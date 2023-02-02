import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
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
      self.h = [] # 运动平面高度
      self.device = torch.device("cpu")   # dataloader只能读取cpu中的数据，如果存到GPU，则dataloader读到的是0
      files = os.listdir(path)
      files = list(filter(file_filter, files))
      self.N = len(files) if size < 0 else size
      for idx in range(0, self.N):
         data = io.loadmat(os.path.join(path, files[idx]), mat_dtype=True)
         # if not np.isnan(data['rmse_imm'][0][0]):
         self.station.append(torch.tensor(data['test_station'][0].reshape((4, 3)), dtype=torch.double, device=self.device))
         self.input.append(torch.tensor(data['test_tdoa'], dtype=torch.double, device=self.device))
         self.target.append(torch.tensor(data['test_data'][:,0:2], dtype=torch.double, device=self.device))
         self.h.append(torch.tensor(data["test_data"][0,2], dtype=torch.double, device=self.device))
         self.rmse_cwls.append(torch.tensor(data['rmse_cwls'], dtype=torch.double, device=self.device))
         self.rmse_imm.append(torch.tensor(data['rmse_imm'], dtype=torch.double, device=self.device))
      self.N = len(self.input)
      self.length = self.input[0].shape[0]

   def __getitem__(self, item):
      return {'z':self.input[item],
              'x':self.target[item],
              'station':self.station[item],
              'rmse_cwls':self.rmse_cwls[item],
              'rmse_imm':self.rmse_imm[item],
              'h':self.h[item]}

   def __len__(self):
      return self.N



class DataloaderList(object):
   def __init__(self):
      self.dataset = self._get_datasets()

   def _get_datasets(self):
      dataloader_list = []
      for dataset_name,dataset_config in config["dataset"].items():
         if dataset_config["is_used"]: # 判断是否使用该数据集
            dataloader = MetaDataLoader(dataset_name, dataset_config)
            dataloader_list.append(dataloader)
      return dataloader_list
   def __getitem__(self, item):
      return self.dataset[item]

class MetaDataLoader():
   # 增加了meta信息的dataloader
   def __init__(self):
      training_set = []
      training_set_path = []
      for dataset_name, dataset_config in config["dataset"].items():
         if dataset_config["is_used"] and dataset_config["is_eval"] == False:
            training_set.append(TDOADataset(dataset_config["path"], size=-1))
            training_set_path.append(dataset_config["path"])
      self.dataset = ConcatDataset(training_set)
      default_dataloader_params = {
         'dataset': self.dataset,
         'batch_size': 1,
         'num_workers': 0,
         'pin_memory': False,
         'drop_last': False,
         'shuffle': False,
         'generator': torch.Generator(device=device)
      }
      self.dataloader_params = default_dataloader_params

