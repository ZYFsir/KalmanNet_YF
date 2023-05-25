import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
import os
import scipy.io as io
import numpy as np
from utils.torchSettings import get_config, get_torch_device


def file_filter(f):
    if f[-4:] in ['.mat']:
        return True
    else:
        return False


class TDOADataset(Dataset):
    def __init__(self, path, name=None, size=-1):
        self.name = name
        # dataloader只能读取cpu中的数据，如果存到GPU，则dataloader读到的是0
        # self.device = torch.device("cpu")
        files = os.listdir(path)
        files = list(filter(file_filter, files))
        self.N = len(files) if size < 0 else size
        self.input = [None] * self.N
        self.target = [None] * self.N
        # self.station = [None] * self.N
        self.rmse_cwls = [None] * self.N
        self.rmse_imm = [None] * self.N
        # self.h = [None] * self.N  # 运动平面高度

        # station 和 h 只读取固定值
        data = io.loadmat(os.path.join(path, files[0]), mat_dtype=False)
        self.station = data['test_station'][0].astype(np.float32)
        self.station.shape = (4, 3)
        self.station = torch.Tensor(self.station)
        self.h = data["test_data"][0, 2].astype(np.float32)
        self.h = torch.as_tensor(self.h)

        data = torch.load("Dataset/trainset.pt")
        self.input = data["input"]
        N, T, m = self.input.shape
        self.target = data["target"]
        # for idx in range(0, self.N):
        #     data = io.loadmat(os.path.join(path, files[idx]), mat_dtype=False)
        #     if idx == 0:
        #         T, m = data['test_tdoa'].shape
        #         self.input = np.zeros((self.N, T, m), dtype="int32")
        #         self.target = np.zeros((self.N, T, 2), dtype="int32")
        #     self.input[idx, :, :] = data['test_tdoa']
        #     self.target[idx, :, :] = data["test_data"][0, 2]
            # if not np.isnan(data['rmse_imm'][0][0]):
            # self.station[idx] = torch.tensor(data['test_station'][0].reshape(
            #     (4, 3)), dtype=torch.float32, device=self.device)
            # self.input[idx] = torch.tensor(
            #     data['test_tdoa'], dtype=torch.float32, device=self.device)
            # self.target[idx] = torch.tensor(
            #     data['test_data'][:, 0:2], dtype=torch.float32, device=self.device)
            # self.h[idx] = torch.tensor(
            #     data["test_data"][0, 2], dtype=torch.float32, device=self.device)
            # self.rmse_cwls.append(torch.tensor(data['rmse_cwls'], dtype=torch.double, device=self.device))
            # self.rmse_imm.append(torch.tensor(data['rmse_imm'], dtype=torch.double, device=self.device))
        self.N = N
        self.length = T

    def __getitem__(self, item):
        return {'z': torch.Tensor(self.input[item, :, :]),
                'x': torch.Tensor(self.target[item, :, :])}
                # 'station': torch.Tensor(self.station),
                # 'rmse_cwls':self.rmse_cwls[item],
                # 'rmse_imm':self.rmse_imm[item],
                # 'h': torch.as_tensor(self.h)}

    def __len__(self):
        return self.N

# class DataloaderList(object):
#     def __init__(self):
#         self.dataset = self._get_datasets()

#     def _get_datasets(self):
#         dataloader_list = []
#         for dataset_name, dataset_config in config["dataset"].items():
#             if dataset_config["is_used"]:  # 判断是否使用该数据集
#                 dataloader = MetaDataLoader(dataset_name, dataset_config)
#                 dataloader_list.append(dataloader)
#         return dataloader_list

#     def __getitem__(self, item):
#         return self.dataset[item]


# class MetaDataLoader():
#     # 增加了meta信息的dataloader

#     def __init__(self):
#         training_set = []
#         training_set_path = []
#         for dataset_name, dataset_config in config["dataset"].items():
#             if dataset_config["is_used"] and dataset_config["is_eval"] == False:
#                 training_set.append(TDOADataset(
#                     dataset_config["path"], size=-1))
#                 training_set_path.append(dataset_config["path"])
#         self.dataset = Dataset(training_set)
#         default_dataloader_params = {
#             'dataset': self.dataset,
#             'batch_size': 1,
#             'num_workers': 0,
#             'pin_memory': False,
#             'drop_last': False,
#             'shuffle': False,
#             'generator': torch.Generator(device=device)
#         }
#         default_dataloader_params.update(config["dataloader_params"])
#         self.dataloader_params = default_dataloader_params
