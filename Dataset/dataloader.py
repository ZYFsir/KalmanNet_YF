import torch
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as io
import numpy as np
from utils.torchSettings import get_config, get_torch_device
from utils import config, device, logger

def file_filter(f):
   if f[-4:] in ['.mat']:
      return True
   else:
      return False

class TDOADataset(Dataset):
   def __init__(self, path, name=None, size=-1):
      self.name = name
      self.input = []
      self.target = []
      self.station = []
      self.rmse_cwls = []
      self.rmse_imm = []
      self.device = get_torch_device()
      files = os.listdir(path)
      files = list(filter(file_filter, files))
      self.N = len(files) if size < 0 else size
      for idx in range(0, self.N):
         data = io.loadmat(os.path.join(path, files[idx]))
         if not np.isnan(data['rmse_imm'][0][0]):
            self.station.append(torch.tensor(data['test_station'][0].reshape((4, 3)), dtype=torch.float, device=self.device))
            self.input.append(torch.tensor(data['test_tdoa'], dtype=torch.float, device=self.device))
            self.target.append(torch.tensor(data['test_data'], dtype=torch.float, device=self.device))

            self.rmse_cwls.append(torch.tensor(data['rmse_cwls'], dtype=torch.float, device=self.device))
            self.rmse_imm.append(torch.tensor(data['rmse_imm'], dtype=torch.float, device=self.device))
      self.N = len(self.input)

      self.length = self.input[0].shape[0]

   def __getitem__(self, item):
      return {'input':self.input[item],
              'target':self.target[item],
              'station':self.station[item],
              'rmse_cwls':self.rmse_cwls[item],
              'rmse_imm':self.rmse_imm[item],}

   def __len__(self):
      return self.N

class DataloaderList():
   def __init__(self):
      self.dataset = self._get_datasets()

   def _get_datasets(self):
      dataloader_list = []
      for dataset_name,dataset_config in config["dataset"].items():
         if dataset_config["is_used"]: # 判断是否使用该数据集
            dataloader = MetaDataLoader(dataset_name, dataset_config)
            dataloader_list.append(dataloader)
      return dataloader_list

class MetaDataLoader():
   # 增加了meta信息的dataloader
   def __init__(self, dataset_name, dataset_config):
      self.name = dataset_name
      self.path = dataset_config["path"]
      self.is_eval = dataset_config["is_eval"]
      default_dataloader_params = {
         'dataset': TDOADataset(self.path, size=-1),
         'batch_size': 1,
         'num_workers': 1,
         'pin_memory': False,
         'drop_last': False,
         'shuffle': True,
         'generator': torch.Generator(device=device)
      }
      if "dataloader_params" in dataset_config:
         self.dataloader_params = {**default_dataloader_params, **dataset_config["dataloader_params"]}   # 字典合并，并覆盖参数
      else:
         self.dataloader_params = default_dataloader_params
      self.dataloader = DataLoader(**self.dataloader_params)
