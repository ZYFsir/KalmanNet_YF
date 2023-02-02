import numpy as np
import os, sys, argparse, torch, numpy

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.dataloader import MetaDataLoader
from utils import logger, device, config

from Filter.EKF import ExtendedKalmanFilter
from Filter.singer_EKF import init_SingerModel
from Filter.KalmanNet import KalmanNet
from UI.analyze import analyze

torch.set_default_dtype(torch.double)


if __name__ == "__main__":
   torch.set_printoptions(precision=12)
   print_now_time()
   dataloader = DataLoader(**(MetaDataLoader().dataloader_params))

   ## using KalmanNet
   ekf = KalmanNet()
   optimizer = torch.optim.Adam(ekf.parameters(), lr=1e-3)
   loss_fn = nn.MSELoss()
   epoch = 2
   not_init_model = True
   ekf.train()

   MSE_epoch = []
   for epoch_i in range(epoch):
      print("****** epoch",epoch_i," *********")
      MSE_testset_singledata = []
      MSE_dB_testset_singledata = []
      for data in dataloader:
         if not_init_model:
            singer_model = init_SingerModel(data["station"], data["h"])
            not_init_model = False
            ekf.set_ssmodel(singer_model)
         ## using EKF
         # ekf = ExtendedKalmanFilter(singer_model)

         ekf.init_hidden()
         (batch_size, T, n) = data["z"].shape
         ekf.model.InitSequence(singer_model.m1x_0, T)

         m = config["dim"]["state"]
         n = config["dim"]["measurement"]

         x_ekf = torch.empty([batch_size, T, m])

         x_true = data["x"].cuda()
         z = data["z"].cuda()
         for t in range(0, T):
            m1x_posterior = ekf.forward(z[:,t,:])
            x_ekf[:, t, :] = m1x_posterior.squeeze(2)

         ## training KalmanNet
         loss = loss_fn(x_true, x_ekf[:, :, 0:2])

         MSE_testset_singledata.append(torch.mean(loss))
         MSE_dB_testset_singledata.append(10 * torch.log10(torch.mean(loss)))
         print("MSE:", torch.mean(loss).item())
         print("MSE(dB):", 10 * torch.log10(torch.mean(loss)).item())

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
      MSE_testset_mean = torch.mean(torch.Tensor(MSE_testset_singledata))
      MSE_dB_testset_mean = torch.mean(torch.Tensor(MSE_dB_testset_singledata))
      print("current epoch MSE:", MSE_testset_mean)
      print("current epoch MSE(dB):", MSE_dB_testset_mean)
   torch.save(ekf.state_dict(), "testmodel.pkl")
   #analyze(ekf, x_ekf.detach(), x_true)


