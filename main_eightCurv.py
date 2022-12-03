import numpy as np
import os, sys, argparse, torch, numpy

import matplotlib.pyplot as plt

from Pipeline_EKF_yf import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from KalmanNet_sysmdl import SystemModel
from EKF_test_yf import EKFTest

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.dataloader import DataloaderList
from utils import logger, device, config

print_now_time()
datasets = DataloaderList()

m = 3    # 输入维度
n = 6    # 输出维度

def init_SystemModel(station, dim, z):
   z = torch.tensor(z, dtype=torch.float, device=device)

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
