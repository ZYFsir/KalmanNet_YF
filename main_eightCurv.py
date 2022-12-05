import numpy as np
import os, sys, argparse, torch, numpy

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.dataloader import MetaDataLoader
from utils import logger, device, config
from Filter.SystemModel import init_SingerModel
from Filter.EKF import ExtendedKalmanFilter
from UI.analyze import analyze

torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
   print_now_time()
   dataloader = DataLoader(**(MetaDataLoader().dataloader_params))
   for data in dataloader:
      singer_model = init_SingerModel(data["station"],data["h"])
      ekf = ExtendedKalmanFilter(singer_model, data["x"])
      x_ekf = ekf.forward(data["z"].cuda()).cpu()
      x_true = data["x"]
      analyze(ekf, x_ekf, x_true)





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
