import torch
import os
from src.config.config import Config
from src.dataloader.TDOADataset import TDOADataset
import matplotlib.pyplot as plt
# 本文件用于研究如何进行归一化

def norm(tdoa, station_distance, min_theta):
    theta = torch.arccos(tdoa/station_distance)
    norm_x = torch.tan(min_theta)/torch.tan(theta)
    return norm_x

if __name__ == "__main__":
    station_i = 2
    os.chdir('../..')
    config = Config('src/config/exp01_kalmanNet.yaml')
    dataset = TDOADataset(config.dataset['train'])
    tdoa = dataset.x[:,:,station_i-1]

    dis_fn = torch.nn.PairwiseDistance(p=2)
    distance = dis_fn(dataset.station[0,:], dataset.station[station_i,:])

    min_theta = torch.Tensor([2.5 / 180]) * torch.pi
    norm_x=norm(tdoa, distance, min_theta).reshape(-1)
    plt.hist(norm_x, bins=60)
    plt.show()
    print('hh')

