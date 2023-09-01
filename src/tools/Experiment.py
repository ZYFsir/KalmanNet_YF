import os
import torch
from torch import nn
from src.tools.torchSettings import get_torch_device
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.dataloader.TDOADataset import TDOADataset
from src.model.KalmanNet import KalmanNet
from src.model.singer_EKF import init_SingerModel
from src.model.EKF import ExtendedKalmanFilter
from torch.utils.tensorboard import SummaryWriter
from src.tools.TBPTT import TBPTT
from src.tools.utils import expand_to_batch, move_to_cuda

def exist_trial(trial_folder):
    model_folder = os.path.join(trial_folder, "model_backups")
    if os.path.exists(model_folder):
        files = os.listdir(model_folder)
        checkpoint_files = [x for x in files if ".pt" in x]
        if len(checkpoint_files) > 0:
            return True
        else:
            return False
    else:
        return False


def mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def get_datasize_from_dataloader(dataloader):
    return dataloader.dataset.N

def get_batchsize_from_dataloader(dataloader):
    return dataloader.batch_size


class Experiment:
    def __init__(self, config):
        self.max_checkpoint_num = 5
        self.experiments_root = "experiments"
        self.sub_folders = {
            "model_backups": None,
            "results": None,
            "tensorboard logs": None
        }

        self.config = config
        self.device = get_torch_device()
        self.loss_function = nn.MSELoss()

        self.model, self.optimizer = self._initialize_model_and_optimizer()
        self.trial_folder = self._initialize_trial_folder()
        self.logger = self._initialize_logger()




    def _initialize_model_and_optimizer(self):
        model = self.create_model_by_name(self.config.model_name)
        optimizer = self.create_optimizer_by_name(self.config.optimizer_name, model.parameters())
        return model, optimizer

    def _initialize_trial_folder(self):
        trial_folder = os.path.join(self.experiments_root, self.config.experiment_name, self.config.trial_name)
        self.create_experiment_log_dir(self.config.experiment_name, self.config.trial_name)
        if exist_trial(trial_folder):
            self.load_checkpoints(self.sub_folders["model_backups"])
        else:
            self.experiment_start_epoch = 1
        return trial_folder

    def _initialize_logger(self):
        return self.get_logger(self.sub_folders["tensorboard logs"])

    def get_dataloader(self, dataset_name):
        if dataset_name in self.config.dataset:
            dataset = TDOADataset(self.config.dataset[dataset_name])
            dataloader = DataLoader(dataset, self.config.batch_size, num_workers=4, drop_last=True)
            return dataloader
        else:
            print("dataset name invalid")
            raise


    def load_checkpoints(self, folder):
        files = os.listdir(folder)
        checkpoint_files = [x for x in files if ".pt" in x]
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[2]))
        checkpoint_rewind = torch.load(
            os.path.join(folder, checkpoint_files[0]), map_location=self.device)
        self.experiment_start_epoch = checkpoint_rewind["epoch"]
        self.model.load_state_dict(checkpoint_rewind["model_state_dict"])
        # self.optimizer.load_state_dict(
        #    checkpoint_rewind["optimizer_state_dict"])


    def train(self, dataloader):
        num_epochs = self.config.epoch
        with tqdm(range(self.experiment_start_epoch, num_epochs + 1), desc="Training", total=num_epochs) as progess_bar:
            progess_bar.update(self.experiment_start_epoch)
            for epoch in range(self.experiment_start_epoch, num_epochs + 1):
                # 一次训练，返回的loss尺寸未经任何压缩,[batchsize, T-length, xy]
                loss = self.train_one_epoch(dataloader)

                # 记录结果
                self.logger.add_scalar("MSE per epoch", loss, epoch)

                # 保存模型
                if epoch % self.config.save_every_epoch == 0:
                    self.save_checkpoint(dataloader, epoch, loss)

                # 更新进度条
                progess_bar.update()
                progess_bar.set_postfix({"loss":loss})

    def save_checkpoint(self, dataloader, epoch_i, loss_train):
        checkpoint_name = os.path.join(self.sub_folders["model_backups"], f"KalmanNet_train_{loss_train}_epoch_{epoch_i}.pt")
        checkpoint_content = {
            'epoch': epoch_i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'MSE_train': loss_train,
        }
        self.validate_checkpoint_nums()
        torch.save(checkpoint_content, checkpoint_name)

    def test(self, dataset_name):
        self.model.eval()
        if dataset_name not in self.dataloader_dict:
            print(f"Exp datase_name {dataset_name} invalid")
            raise
        trajs_num = self.dataset_size_dict[dataset_name] // self.batch_size * self.batch_size
        T = self.dataloader_dict[dataset_name].dataset.length
        loss = torch.empty([trajs_num, T, 2])

        with torch.no_grad():
            for data_i, data in enumerate(self.dataloader_dict[dataset_name]):
                if data_i == 0:
                    singer_model = init_SingerModel(
                        data["station"], data["h"], self.m, self.n, self.config["Observation model"]["r"], self.device)
                    self.model.set_ssmodel(singer_model)
                    self.model.to(self.device)
                # 基本变量读取
                x_ekf = torch.empty([self.batch_size, T, self.n])
                x_true = data["x"].cuda()
                z = data["z"].cuda()
                if self.model_name == "KNet":
                    self.model.init_hidden()
                    self.model.InitSequence()
                    for t in range(0, T):
                        m1x_posterior = self.model.forward(z[:, t, :])
                        x_ekf[:, t, :] = m1x_posterior.squeeze(2)[:, 0:3]
                elif self.model_name == "EKF":
                    self.model.InitSequence(singer_model.m1x_0, singer_model.m2x_0)
                    x_ekf = self.model.forward(z)

                # 求loss
                loss_points = self.loss_function(x_true, x_ekf[:, :, 0:2])
                loss[data_i * self.batch_size:(data_i + 1) * self.batch_size, :, :] = loss_points
            # 记录结果
            print("test loss:", torch.mean(loss))
            torch.save(loss, f"Result/test_loss/{self.model_name}_{dataset_name}_loss.pt")

    def train_one_epoch(self, dataloader):
        self.model.train()

        total_loss = 0
        for data_i, batch in enumerate(dataloader):
            batch = move_to_cuda(batch, self.device)
            inputs, targets, station, h = batch
            h = torch.Tensor(h).cuda()
            inputs = inputs.transpose(1,2).contiguous()
            targets = targets.transpose(1,2).contiguous()
            batch_size = inputs.size(0)

            self.optimizer.zero_grad()
            accumulated_loss = 0
            hidden_states = None
            self.initialize_model_beliefs(batch_size)
            backward_sequence_length = self.config.backward_sequence_length
            for t in range(0, inputs.size(1), backward_sequence_length):
                outputs, hidden_states = self.model(inputs[:,:, t:t + backward_sequence_length], hidden_states, station, h)
                loss = self.loss_function(outputs[:,0:2,:], targets[:,:,t:t + backward_sequence_length])
                loss.backward(retain_graph = True)
                accumulated_loss += loss

            self.optimizer.step()

            total_loss += accumulated_loss.item()

        return total_loss / len(dataloader)

        #
        #     # 基本变量读取
        #     (batch_size, T, n) = y.shape
        #
        #     x = x.cuda()
        #     y = y.cuda()
        #     self.model.InitSequence()
        #     y_net_output = torch.empty([batch_size, T, self.config.target_state_dim])
        #     hidden = self.model.get_init_hidden()
        #
        #     for t in range(0, T):
        #         m1x_posterior, hidden = self.model.forward(x[:, t, :], hidden)
        #         # print("KG loss:", loss.item())
        #
        #         y_net_output[:, t, :] = m1x_posterior.squeeze(2)
        #     # 求loss
        #     loss = self.loss_function(y, y_net_output[:, :, 0:2])
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     # loss_trajs_in_iter = torch.mean(
        #     #     loss_elements_in_iter, dim=(1, 2))
        #     # loss_batch_in_iter = torch.mean(loss_trajs_in_iter)
        #
        #     # 更新网络权重
        #     # self.scheduler.step()
        # return loss

    def initialize_model_beliefs(self, batch_size):
        mean = torch.Tensor(self.config.init_state_mean)
        covariance = self.config.init_state_covariance_scaling * torch.eye(self.config.target_state_dim)

        # Initialize the sequence in the model
        mean = expand_to_batch(mean, batch_size)
        covariance = expand_to_batch(covariance, batch_size)
        self.model.initialize_beliefs(mean, covariance)


    def validate_checkpoint_nums(self):
        # 计算当前的模型数量
        checkpoint_count = 0
        files = os.listdir(
            self.sub_folders["model_backups"])
        checkpoint_files = [
            x for x in files if "KalmanNet.pt" in x]
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[0]))
        checkpoint_count = len(checkpoint_files)
        # 清除过多保存的模型
        if checkpoint_count > self.max_checkpoint_num:
            for i in range(0, checkpoint_count - self.max_checkpoint_num):
                remove_checkpoint_name = os.path.join(self.sub_folders["model_backups"], checkpoint_files[-1])
                if os.path.exists(remove_checkpoint_name):
                    os.remove(remove_checkpoint_name)
                else:
                    print("要删除的文件不存在！")

    def create_model_by_name(self, model_name):
        if model_name == "KalmanNet":
            model = KalmanNet(self.config.in_mult, self.config.out_mult,
                              self.config.target_state_dim, self.config.measurement_dim, self.device)
        else:
            print("model_name is invalid")
            raise
        return model

    def create_optimizer_by_name(self, name, parameters):
        if name == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=self.config.learning_rate)
        elif name == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        else:
            print("optimizer_name is invalid")
            raise
        return optimizer

    def create_experiment_log_dir(self, experiment_name, trial_name):
        trial_dir = os.path.join(self.experiments_root, experiment_name, trial_name)
        for key in self.sub_folders.keys():
            self.sub_folders[key] = os.path.join(trial_dir, key)
            mkdir(self.sub_folders[key])

    def get_logger(self, path):
        return SummaryWriter(path)
