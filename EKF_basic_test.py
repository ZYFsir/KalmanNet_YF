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

from main_eightCurv import init_SystemModel, init_dataloaders, TDOADataset

batch_size = 4
train, cv, test, T_train, T_cv, T_test, station, z = init_dataloaders(batch_size)

singer_model = init_SystemModel(station, dim=2, z=z)
n = 9
m1x_0 = torch.zeros(n, 1)
m1x_0[0:3] = 1e4+m1x_0[0:3]
m2x_0 = 1 * torch.eye(n)
singer_model.InitSequence(m1x_0, m2x_0)

from EKF_yf import ExtendedKalmanFilter
EKF = ExtendedKalmanFilter(singer_model, 'full')
