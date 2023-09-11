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
from src.tools.utils import expand_to_batch, move_to_device, state_detach

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
            self.load_checkpoints(self.sub_folders["model_backups"], self.config.checkpoint_name)
        else:
            self.experiment_start_epoch = 1
        return trial_folder

    def _initialize_logger(self):
        return self.get_logger(self.sub_folders["tensorboard logs"])

    def get_dataloader(self, dataset_name):
        if dataset_name in self.config.dataset:
            dataset = TDOADataset(self.config.dataset[dataset_name])
            dataloader = DataLoader(dataset, self.config.batch_size, num_workers=4)
            dataloader.station = dataset.station
            dataloader.h = dataset.h
            return dataloader
        else:
            print("dataset name invalid")
            raise


    def load_checkpoints(self, folder, file_name):
        checkpoint_rewind = torch.load(
            os.path.join(folder, file_name), map_location=self.device)
        self.experiment_start_epoch = checkpoint_rewind["epoch"]
        self.model.load_state_dict(checkpoint_rewind["model_state_dict"])
        # self.optimizer.load_state_dict(
        #    checkpoint_rewind["optimizer_state_dict"])


    def train(self, dataset_name):
        dataloader = self.get_dataloader(dataset_name)

        num_epochs = self.config.epoch
        if self.config.use_scheduler is True:
            self.scheduler = self.initialize_scheduler(self.optimizer, self.config.scheduler_name, num_epochs+1)
        with tqdm(range(self.experiment_start_epoch, num_epochs + 1), desc="Epochs", leave=True,total=num_epochs) as progess_bar:
            progess_bar.update(self.experiment_start_epoch)
            for epoch in range(self.experiment_start_epoch, num_epochs + 1):
                # 一次训练，返回的loss尺寸未经任何压缩,[batchsize, T-length, xy]
                loss = self.train_one_epoch(dataloader)

                if self.config.use_scheduler is True:
                    self.scheduler.step(loss)
                # 记录结果
                self.logger.add_scalar("MSE per epoch", loss, epoch)

                # 保存模型
                if epoch % self.config.save_every_epoch == 0:
                    self.save_checkpoint(epoch, loss)

                # 更新进度条
                progess_bar.update()
                progess_bar.set_postfix({"loss":loss})

    def save_checkpoint(self, epoch_i, loss_train):
        checkpoint_name = os.path.join(self.sub_folders["model_backups"], f"KalmanNet_train_{loss_train}_epoch_{epoch_i}.pt")
        checkpoint_content = {
            'epoch': epoch_i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'MSE_train': loss_train,
        }
        self.validate_checkpoint_nums()
        torch.save(checkpoint_content, checkpoint_name)

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        datasize = get_datasize_from_dataloader(dataloader)
        station = move_to_device(dataloader.station, self.device)
        h = move_to_device(dataloader.h, self.device)
        with tqdm(range(datasize), desc="dataset", position=1) as data_progress_bar:
            with torch.no_grad():
                for data_i, batch in enumerate(dataloader):
                    batch = move_to_device(batch, self.device)
                    inputs, targets = batch

                    batch_size = inputs.size(0)
                    hidden_in_states = None

                    self.initialize_model_beliefs(batch_size)

                    hidden_list = []
                    hidden_list.append(hidden_in_states)
                    output_sequence = torch.empty([batch_size, 2, inputs.size(2)])
                    for t in range(0, inputs.size(2)):
                        outputs, hidden_out_states = self.model(inputs[:, :, t:t + 1], hidden_in_states,
                                                                station, h)
                        hidden_in_states = hidden_out_states
                        output_sequence[:,:,t:t+1] = outputs[:,0:2,:]
                    loss = self.loss_function(output_sequence, targets)
                    average_batch_loss = loss / (2 * batch_size * inputs.size(2))

                    data_progress_bar.set_postfix({"loss": average_batch_loss})
                    data_progress_bar.update(batch_size)

                    total_loss += average_batch_loss
        return total_loss / (data_i + 1)



    def train_one_epoch(self, dataloader):
        self.model.train()

        total_loss = 0
        datasize = get_datasize_from_dataloader(dataloader)
        station = move_to_device(dataloader.station, self.device)
        h = move_to_device(dataloader.h, self.device)
        with tqdm(range(datasize), desc="dataset", position=1) as data_progress_bar:
            for data_i, batch in enumerate(dataloader):
                batch = move_to_device(batch, self.device)
                inputs, targets = batch

                batch_size = inputs.size(0)

                self.optimizer.zero_grad()
                accumulated_loss = 0
                hidden_in_states = None
                with torch.no_grad():
                    self.initialize_model_beliefs(batch_size)

                hidden_list = []
                hidden_list.append(hidden_in_states)
                single_data_loss = 0
                for t in range(0, inputs.size(2)):
                    outputs, hidden_out_states = self.model(inputs[:, :, t:t + 1], hidden_in_states,
                                                            station, h)
                    hidden_list.append(hidden_out_states)
                    loss = self.loss_function(outputs[:, 0:2, :], targets[:, :, t:t + 1])
                    accumulated_loss = accumulated_loss + loss  # 此处不可用+=，因为inplace操作会影响计算图
                    single_data_loss = single_data_loss + loss.item()

                    if (t + 1) % self.config.backward_sequence_length == 0:
                        # Perform a backward pass and update gradients every 10 time steps
                        self.optimizer.zero_grad()
                        accumulated_loss.backward()
                        self.optimizer.step()
                        accumulated_loss = 0
                        hidden_in_states = state_detach(hidden_list[-1])
                    else:
                        hidden_in_states = hidden_list[-1]

                # Perform a final backward pass and update gradients for any remaining accumulated loss
                if accumulated_loss.item() > 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    self.optimizer.step()
                average_single_data_loss = single_data_loss/(2*batch_size*inputs.size(2))
                data_progress_bar.set_postfix({"loss":average_single_data_loss})
                data_progress_bar.update(batch_size)
                total_loss = total_loss + average_single_data_loss
        return total_loss / (data_i+1)

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
            x for x in files if "KalmanNet" in x]
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[2]))
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
                              self.config.target_state_dim, self.config.measurement_dim, self.device, self.config.kalman_gain_model)
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

    def initialize_scheduler(self, optimizer, scheduler_name, total_epochs):
        if scheduler_name == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer,verbose=True)
        else:
            print("invalid scheduler name")
            raise
        return scheduler
