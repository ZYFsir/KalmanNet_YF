import numpy as np
import os
import sys
import argparse
import torch
import numpy

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.TDOADataset import TDOADataset
from utils.Experiment import Experiment

from Filter.EKF import ExtendedKalmanFilter
from Filter.singer_EKF import init_SingerModel
from Filter.KalmanNet import KalmanNet
from UI.analyze import analyze

from src.config.config import Config

torch.set_printoptions(precision=12)

if __name__ == "__main__":
    print_now_time()

    config = Config('src/config/exp01_kalmanNet.yaml')
    experiment = Experiment(config)
    # experiment.run(mode="test", dataset_name="train")
    experiment.test("train")

