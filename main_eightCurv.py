from datetime import datetime

import numpy as np
import torch
from torch import optim
import os, sys, argparse, torch, numpy
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import scipy.io as io
from Pipeline_EKF_yf import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from KalmanNet_sysmdl import SystemModel
from EKF_test_yf import EKFTest

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

##################
### Dataloader ###
##################
def init_dataloaders(batch_size=1, num_workers=0):
   dataset_train = TDOADataset("Simulations/MatlabSet/train", size=-1)
   dataset_cv = TDOADataset("Simulations/MatlabSet/cv", size=-1)
   dataset_test = TDOADataset("Simulations/MatlabSet/test", size=-1)

   T_train = dataset_train.length
   T_cv = dataset_cv.length
   T_test = dataset_test.length

   print("读取训练集：Size  ", dataset_train.N, "  T  ",dataset_train.length)
   print("读取验证集：Size  ", dataset_cv.N, "  T  ",dataset_cv.length)
   print("读取测试集：Size  ", dataset_test.N, "  T  ",dataset_test.length)
   dataloader_train_params = {
      'dataset': dataset_train,
      'batch_size': batch_size,
      'num_workers': num_workers,
      'pin_memory': False,
      'drop_last': False,
      'shuffle': True,
      'generator':torch.Generator(device=cuda0)
   }
   dataloader_cv_params = {
      'dataset': dataset_cv,
      'batch_size': batch_size,
      'num_workers': num_workers,
      'pin_memory': False,
      'drop_last': False,
      'shuffle': True,
      'generator': torch.Generator(device=cuda0)
   }
   dataloader_test_params = {
      'dataset': dataset_test,
      'batch_size': batch_size,
      'num_workers': num_workers,
      'pin_memory': False,
      'drop_last': False,
      'shuffle': True,
      'generator': torch.Generator(device=cuda0)
   }
   dataloader_train = DataLoader(**dataloader_train_params)
   dataloader_cv = DataLoader(**dataloader_cv_params)
   dataloader_test = DataLoader(**dataloader_test_params)
   # return dataloader_test
   station = dataset_train[0]['station']
   z = 10000
   return dataloader_train, dataloader_cv, dataloader_test, T_train, T_cv, T_test, station, z

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

      files = os.listdir(path)
      files = list(filter(file_filter, files))
      self.N = len(files) if size < 0 else size
      for idx in range(0, self.N):
         data = io.loadmat(os.path.join(path, files[idx]))
         if not numpy.isnan(data['rmse_imm'][0][0]):
            self.station.append(torch.tensor(data['test_station'][0].reshape((4, 3)), dtype=torch.float, device=cuda0))
            self.input.append(torch.tensor(data['test_tdoa'],  dtype=torch.float, device=cuda0))
            self.target.append(torch.tensor(data['test_data'], dtype=torch.float, device=cuda0))

            self.rmse_cwls.append(torch.tensor(data['rmse_cwls'], dtype=torch.float, device=cuda0))
            self.rmse_imm.append(torch.tensor(data['rmse_imm'], dtype=torch.float, device=cuda0))
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

batch_size = 4
train, cv, test, T_train, T_cv, T_test, station, z = init_dataloaders(batch_size)

##################
###  SysModel  ###
##################
m = 3    # 输入维度
n = 6    # 输出维度

def init_SystemModel(station, dim, z):
   z = torch.tensor(z, dtype=torch.float, device=cuda0)

   q = 1e4   # 模型噪声
   r = 1e-3    # 测量噪声

   I = torch.eye(2, dtype=torch.float,device=torch.device("cuda:0"))
   tao = 1
   alpha = 0.1
   def f(x):
      F1 = torch.hstack((I, tao * I, (alpha*tao-1+np.exp(-alpha*tao)) / (alpha ** 2) * I))
      # F1[2,:] = torch.tensor([0,0,1,0,0,0,0,0,0])
      F2 = torch.hstack((0*I, I, (1-np.exp(-alpha*tao))/alpha*I))
      # F2[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
      F3 = torch.hstack((0*I, 0*I,  np.exp(-alpha*tao)*I))
      # F3[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
      F = torch.vstack((F1, F2, F3))
      return torch.matmul(F, x).squeeze()


   def h(x):
      u = torch.hstack((x[0:2], z))
      r = torch.norm(u - station, dim = 1)
      tdoa = torch.abs(r[1:] - r[0])
      return tdoa.squeeze()
   Q_true = (q**2) * torch.eye(n)
   R_true = (r**2) * torch.eye(m)

   T = 100
   T_test = 152
   return SystemModel(f, Q_true, h, R_true, T, T_test)

# for train_data in train:
#    station = train_data['station']
#    input = train_data['input']
#    target = train_data['target']
singer_model = init_SystemModel(station, dim=2, z=z)
m1x_0 = torch.zeros(n, 1)
m1x_0[0:3,0] = torch.tensor([13456, 104091, 10000])
m2x_0 = 1e6 * torch.eye(n)
singer_model.InitSequence(m1x_0, m2x_0)

[MSE_EKF_linear_arr_partialoptq, MSE_EKF_linear_avg_partialoptq, MSE_EKF_dB_avg_partialoptq, EKF_out_partialoptq] = EKFTest(singer_model, test, T_test                                                                                                                                             ,batch_size)
print("ekf finished")

# plt.plot(MSE_EKF_linear_arr_partialoptq.reshape([-1,1]).cpu())
plt.legend(["先验结果", "后验估计结果"])
plt.show()


# modelFolder = 'KNet' + '/'
# KNet_Pipeline = Pipeline_EKF(strTime, "KNet_yf", "KalmanNet")
# KNet_Pipeline.setssModel(singer_model)
# KNet_model = KalmanNetNN()
# KNet_model.Build(singer_model)
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setTrainingParams(n_Epochs=2, n_Batch=100, learningRate=5e-3, weightDecay=1e-4)
#
# KNet_Pipeline.NNTrain(train, cv, test)
