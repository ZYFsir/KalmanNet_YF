import numpy as np
import os, sys, argparse, torch, numpy

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.dataloader import MetaDataLoader
from utils import logger, device, config
from Filter.SystemModel import init_SingerModel
from Filter.EKF import ExtendedKalmanFilter

if __name__ == "__main__":
   print_now_time()
   dataloader = DataLoader(**(MetaDataLoader().dataloader_params))
   for data in dataloader:
      singer_model = init_SingerModel(data["station"],data["h"])
      ekf = ExtendedKalmanFilter(singer_model)
      x_ekf = ekf.forward(data["z"])
      x_true = data["x"]
      analyze(ekf, x_ekf, x_true)



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
