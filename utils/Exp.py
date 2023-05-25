import os
import time
import torch
from comet_ml import Experiment
from torch import nn
from torch.utils.data import DataLoader

from Dataset.TDOADataset import TDOADataset
from Filter.KalmanNet import KalmanNet
from Filter.singer_EKF import init_SingerModel
from utils.torchSettings import get_torch_device, get_config
from torch.profiler import profile, record_function, ProfilerActivity

class Exp:
    def __init__(self, model_name="KNet"):
        self.device = get_torch_device()
        self.config = get_config()
        self.model_name = model_name

        # config变量读取
        self.m = self.config["dim"]["state"]
        self.n = self.config["dim"]["measurement"]
        self.batch_size = self.config["dataloader_params"]["batch_size"]
        self.max_checkpoint_num = self.config["max_checkpoint_num"]
        self.epoch = self.config["training"]["epoch"]
        self.use_comet = self.config["use_comet"]

        # 部分变量初始化

        # 名称与函数的对应字典
        self.optimizer_config_dict = {
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "ASGD": torch.optim.ASGD
        }
        self.scheduler_config_dict = {
            "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau
        }

        self.loss_fn = nn.MSELoss(reduction="none")
        self.log_module()
        self.datasets_module()
        self.model_module(model_name)
        self.train_module()
        self.load_checkpoints()

    def log_module(self):
        # Logger创建
        # init_logger()
        # self.logger = get_logger()
        self.logger = Experiment(
            api_key="0mjX0DZYgTf8rERcUWj7Jo1q1",
            project_name="KalmanNet_Project",
            workspace="zyfsir",
        )
        self.logger.log_parameters(self.config)

    def datasets_module(self):
        # 数据集加载
        self.dataloader_dict = {}
        self.dataset_size_dict = {}
        for dataset_name, dataset_config in self.config["dataset"].items():
            if dataset_config['is_used'] == False:
                continue
            dataset_ = TDOADataset(dataset_config["path"])
            self.station = dataset_.station
            self.h = dataset_.h
            self.dataloader_dict[dataset_name] = DataLoader(
                dataset=dataset_, **self.config["dataloader_params"])
            self.dataset_size_dict[dataset_name] = self.dataloader_dict[dataset_name].dataset.N

    def model_module(self, model_name):
        # 创建模型
        if model_name == "KNet":
            ekf = KalmanNet(self.config["in_mult"], self.config["out_mult"],
                            self.m, self.n, self.batch_size, self.device)
        elif model_name == "EKF":
            pass
            # ekf = ExtendedKalmanFilter(singer_model)
        self.model = ekf.to(self.device)

    def load_checkpoints(self):
        # 读取最优checkpoints
        if not os.path.exists(self.config["checkpoints_saving_folder"]):
            os.makedirs(self.config["checkpoints_saving_folder"])
            return
        files = os.listdir(self.config["checkpoints_saving_folder"])
        checkpoint_files = [x for x in files if "KalmanNet.pt" in x]
        if not checkpoint_files:
            return
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: float(x.split("_")[0]))
        checkpoint_rewind = torch.load(
            os.path.join(self.config["checkpoints_saving_folder"], checkpoint_files[0]), map_location=self.device)
        self.epoch_i = checkpoint_rewind["epoch"]
        self.model.load_state_dict(checkpoint_rewind["model_state_dict"])
        self.optimizer.load_state_dict(
            checkpoint_rewind["optimizer_state_dict"])

    def train_module(self):
        """
        对训练有关的变量、对象进行初始化和构造
        """
        self.epoch_i = 0
        # 初始化optimizer和scheduler
        if "params" in self.config["training"]["optimizer"]:
            self.optimizer = self.optimizer_config_dict[self.config["training"]["optimizer"]["name"]](
                self.model.parameters(), **self.config["training"]["optimizer"]["params"])
        else:
            self.optimizer = self.optimizer_config_dict[self.config["training"]
            ["optimizer"]["name"]](self.model.parameters())
        if "params" in self.config["training"]["scheduler"]:
            self.scheduler = self.scheduler_config_dict[self.config["training"]["scheduler"]["name"]](
                self.optimizer, **self.config["training"]["scheduler"]["params"])
        else:
            self.scheduler = self.scheduler_config_dict[self.config["training"]
            ["scheduler"]["name"]](self.optimizer)

    def run(self, mode="test", dataset_name="test"):
        if mode == "train":
            epoch_i = self.epoch_i
            epoch = self.epoch
            self.model.train()
        elif mode == "test":
            epoch_i = 0
            epoch = 1
            self.model.eval()
        else:
            print(f"Exp run mode {mode} invalid")
            raise
        if dataset_name not in self.dataloader_dict:
            print(f"Exp datase_name {dataset_name} invalid")
            raise
        MSE_per_epoch = torch.empty([self.epoch])

        if self.model_name == "KNet":
            singer_model = init_SingerModel(
                self.station, self.h, self.m, self.n, self.config["Observation model"]["r"], self.device)
            self.model.set_ssmodel(singer_model)
        while epoch_i < epoch:
            print(f"******* epoch {epoch_i} *********")
            start = time.process_time()
            # 一次训练
            MSE_per_batch = self.run_one_epoch(mode=mode, dataset_name=dataset_name)
            print("epoch结束，开始Loss计算")
            # loss计算
            total_loss = torch.mean(MSE_per_batch)
            MSE_per_epoch[epoch_i] = total_loss
            # 记录结果
            # print("计算loss结束，log记录结果")
            # self.logger.log_metrics({
            #     f"MSE_{dataset_name}": MSE_per_epoch[epoch_i],
            #     f"MSE_dB_{dataset_name}": 10 * torch.log10(MSE_per_epoch[epoch_i]),
            # }, epoch=epoch_i)

            # 保存模型
            if mode == "train":
                if epoch_i % 10 == 0:
                    MSE_test = self.run_one_epoch(mode="test", dataset_name="test")
                    loss_test = torch.mean(MSE_test)
                    checkpoint_name = os.path.join(self.config["checkpoints_saving_folder"],
                                                   f"{MSE_per_epoch[epoch_i]}_dB_epoch{epoch_i}_KalmanNet.pt")
                    checkpoint_content = {
                        'epoch': epoch_i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'MSE': MSE_per_epoch[epoch_i],
                        'MSE_test': loss_test,
                    }
                    self.save_checkpoints(name=checkpoint_name, content=checkpoint_content)
            epoch_i += 1
            end = time.process_time()

            print("耗时：",end-start, "s")


    def run_one_epoch(self, mode="test", dataset_name="test"):
        iter_num = self.dataset_size_dict[dataset_name] // self.batch_size
        MSE_per_batch = torch.empty([iter_num])

        for data_i, data in enumerate(self.dataloader_dict[dataset_name]):
            # 基本变量读取
            (batch_size, T, n) = data["z"].shape
            self.model.batch_size = batch_size
            x_ekf = torch.empty([batch_size, T, self.m])

            x_true = data["x"].to(self.device, non_blocking=True)
            z = data["z"].to(self.device, non_blocking=True)
            if self.model_name == "KNet":
                self.model.init_hidden()
                self.model.InitSequence()
                self.model.forward(z)
                # with torch.profiler.profile(
                #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #         #activities=[ProfilerActivity.CUDA],
                #         schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=2),
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./Logs/KalmanNet'),
                #         record_shapes=False,
                #         with_stack=True,
                #         use_cuda=True,
                #         with_modules=True
                #         ) as prof:
                # for t in range(0, T):
                #     m1x_posterior = self.model.forward(z[:, t, :])
                #     x_ekf[:, t, :] = m1x_posterior.squeeze(2)
                        # prof.step()
                # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            elif self.model_name == "EKF":
                # TODO: 此处功能尚未实现
                # ekf.InitSequence(singer_model.m1x_0, singer_model.m2x_0)
                # x_ekf = ekf.forward(z)
                pass
            # 求loss
            # print("开始计算loss")
            loss_elements_in_iter = self.loss_fn(x_true, x_ekf[:, :, 0:2])
            loss_trajs_in_iter = torch.mean(
                loss_elements_in_iter, dim=(1, 2))
            loss_batch_in_iter = torch.mean(loss_trajs_in_iter)
            loss_batch_in_iter.requires_grad_(True)
            # 更新网络权重
            if mode == "train":
                self.optimizer.zero_grad()
                loss_batch_in_iter.backward()
                self.optimizer.step()
                # self.scheduler.step(loss_batch_in_iter)
                self.scheduler.step()
                # print("更新网络权重完成")
            MSE_per_batch[data_i] = loss_batch_in_iter.item()
            # MSE_dB_trainset_singledata[data_i] = 10 * \
            #     torch.log10(torch.mean(loss_element)).item()
            # print(
            #     f"{data_i+1}/{iter_num} MSE:{MSE_per_batch[data_i]}")
            # print(
            #     f"{data_i+1}/{iter_num} MSE(dB):{10*torch.log10(MSE_per_batch[data_i])}")
        return MSE_per_batch

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
        print("准备保存模型")
        torch.save(content, name)
