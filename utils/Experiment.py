import os
import torch
from torch import nn
from utils.torchSettings import get_torch_device, get_config
from torch.utils.data import Dataset, DataLoader

import numpy as np
from Dataset.TDOADataset import TDOADataset
from Filter.KalmanNet import KalmanNet
from Filter.singer_EKF import init_SingerModel
from Filter.EKF import ExtendedKalmanFilter


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


class Experiment:
    def __init__(self, config):
        self.experiments_root = "experiments"
        self.loss_function = nn.MSELoss(reduction="none")

        self.config = config

        self.device = get_torch_device()

        self.model = self.create_model_by_name(self.config.model_name)
        self.optimizer = self.create_optimizer_by_name(self.config.optimizer_name, self.model.parameters())

        self.trial_folder = os.path.join(self.experiments_root, self.config.experiment_name, self.config.trial_name)
        if exist_trial(self.trial_folder):
            self.load_checkpoints(self.trial_folder)
        else:
            self.create_experiment_log_dir(self.config.experiment_name, self.config.trial_name)

    def set_dataloader(self, dataset_name):
        if dataset_name in self.config.dataset:
            dataset = TDOADataset(self.config.dataset[dataset_name])
            dataloader = DataLoader(dataset, self.config.batch_size)
        else:
            print("dataset name invalid")
            raise
    def datasets_module(self):
        # 数据集加载
        self.dataloader_dict = {}
        self.dataset_size_dict = {}
        for dataset_name, dataset_config in self.config["dataset"].items():
            if dataset_config['is_used'] == False:
                continue
            self.dataloader_dict[dataset_name] = DataLoader(
                dataset=TDOADataset(dataset_config["path"]),
                **self.config["dataloader_params"])
            self.dataset_size_dict[dataset_name] = self.dataloader_dict[dataset_name].dataset.N

    def model_module(self, model_name):
        # 创建模型
        if model_name == "KNet":
            ekf = KalmanNet(self.config["in_mult"], self.config["out_mult"],
                            self.m, self.n, self.batch_size, self.device)
        elif model_name == "EKF":
            ekf = ExtendedKalmanFilter()
        self.model = ekf.to(self.device)

    def load_checkpoints(self, folder):
        files = os.listdir(folder)
        checkpoint_files = [x for x in files if ".pt" in x]
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[0]))
        checkpoint_rewind = torch.load(
            os.path.join(folder, checkpoint_files[0]), map_location=self.device)
        self.epoch_i = checkpoint_rewind["epoch"]
        self.model.load_state_dict(checkpoint_rewind["model_state_dict"])
        self.optimizer.load_state_dict(
            checkpoint_rewind["optimizer_state_dict"])

    def train(self):
        epoch_i = self.epoch_i
        epoch = self.epoch
        self.model.train()

        iter_num = self.dataset_size_dict[] // self.batch_size
        MSE_per_epoch = torch.empty([self.epoch])

        while epoch_i < epoch:
            print(f"******* epoch {epoch_i} *********")
            # 一次训练，返回的loss尺寸未经任何压缩,[batchsize, T-length, xy]
            loss_point = self.run_one_epoch(mode=mode, dataset_name="train")
            loss_trajs_in_iter = torch.mean(
                loss_point, dim=(1, 2))  # [batch_size]
            loss = torch.mean(loss_trajs_in_iter)
            MSE_per_epoch[epoch_i] = loss

            # 记录结果
            print("记录结果")
            if mode == "train":
                self.logger.log_metrics({
                    f"MSE_{dataset_name}": MSE_per_epoch[epoch_i],
                    "MSE_dB_{dataset_name}": 10 * torch.log10(MSE_per_epoch[epoch_i]),
                }, epoch=epoch_i)
            elif mode == "test":
                print("test loss:", MSE_per_epoch[epoch_i])

            # 保存模型
            if mode == "train":
                if epoch_i % 5 == 0:
                    MSE_test = self.run_one_epoch(mode="test", dataset_name="test")
                    loss_test = torch.mean(MSE_test)
                    checkpoint_name = self.config["checkpoints_saving_folder"] + \
                                      f"/{MSE_per_epoch[epoch_i]}_dB_epoch{epoch_i}_KalmanNet.pt"
                    checkpoint_content = {
                        'epoch': epoch_i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'MSE': MSE_per_epoch[epoch_i],
                        'MSE_test': loss_test,
                    }
                    self.save_checkpoints(name=checkpoint_name, content=checkpoint_content)
            elif mode == "test":
                pass
            epoch_i += 1

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

    def run_one_epoch(self, mode="test", dataset_name="test"):
        iter_num = self.dataset_size_dict[dataset_name] // self.batch_size
        MSE_per_batch = torch.empty([iter_num])
        for data_i, data in enumerate(self.dataloader_dict[dataset_name]):
            if data_i == 0:
                singer_model = init_SingerModel(
                    data["station"], data["h"], self.m, self.n, self.config["Observation model"]["r"], self.device)
                self.model.set_ssmodel(singer_model)
                self.model.to(self.device)

            # 基本变量读取
            (batch_size, T, n) = data["z"].shape
            self.model.batch_size = batch_size
            x_ekf = torch.empty([batch_size, T, self.m])
            x_true = data["x"].cuda()
            z = data["z"].cuda()
            if self.model_name == "KNet":
                self.model.init_hidden()
                self.model.InitSequence()
                for t in range(0, T):
                    m1x_posterior = self.model.forward(z[:, t, :])
                    x_ekf[:, t, :] = m1x_posterior.squeeze(2)
            elif self.model_name == "EKF":
                self.model.InitSequence(singer_model.m1x_0, singer_model.m2x_0)
                x_ekf = self.model.forward(z)

            # 求loss
            loss_elements_in_iter = self.loss_function(x_true, x_ekf[:, :, 0:2])
            loss_trajs_in_iter = torch.mean(
                loss_elements_in_iter, dim=(1, 2))
            loss_batch_in_iter = torch.mean(loss_trajs_in_iter)

            # 更新网络权重
            if mode == "train":
                self.optimizer.zero_grad()
                loss_batch_in_iter.backward()
                self.optimizer.step()
                # self.scheduler.step(loss_batch_in_iter)
                self.scheduler.step()
            MSE_per_batch[data_i] = loss_batch_in_iter.item()
            # MSE_dB_trainset_singledata[data_i] = 10 * \
            #     torch.log10(torch.mean(loss_element)).item()
            # print(
            #     f"{data_i+1}/{iter_num} MSE:{MSE_per_batch[data_i]}")
            # print(
            #     f"{data_i+1}/{iter_num} MSE(dB):{10*torch.log10(MSE_per_batch[data_i])}")
        return loss_elements_in_iter

    def save_checkpoints(self, name, content):
        # 计算当前的模型数量
        checkpoint_count = 0
        files = os.listdir(
            self.config["checkpoints_saving_folder"])
        checkpoint_files = [
            x for x in files if "KalmanNet.pt" in x]
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[0]))
        checkpoint_count = len(checkpoint_files)
        # 清除过多保存的模型
        if checkpoint_count > self.config["max_checkpoint_num"]:
            for i in range(0, checkpoint_count - self.config["max_checkpoint_num"]):
                remove_checkpoint_name = os.path.join(self.config["checkpoints_saving_folder"], checkpoint_files[-1])
                if os.path.exists(remove_checkpoint_name):
                    os.remove(remove_checkpoint_name)
                else:
                    print("要删除的文件不存在！")
        print("保存模型")
        torch.save(content, name)

    def create_model_by_name(self, model_name):
        if model_name == "KalmanNet":
            model = KalmanNet(self.config.in_mult, self.config.out_mult,
                              self.config.target_state_dim, self.config.measurement_dim, self.config.batch_size,
                              self.device)
        else:
            print("model_name is invalid")
            raise
        return model

    def create_optimizer_by_name(self, name, parameters):
        if name == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=1e-5)
        else:
            print("optimizer_name is invalid")
            raise

    def create_experiment_log_dir(self, experiment_name, trial_name):

        trial_dir = os.path.join(self.experiments_root, experiment_name, trial_name)

        sub_folders = {"model_backups":None, "results":None, "tensorboard logs":None}
        for key in sub_folders.keys():
            sub_folders[key] = os.path.join(trial_dir, key)
            mkdir(sub_folders[key])

