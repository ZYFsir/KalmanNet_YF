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
from utils.Exp import Exp

from Filter.EKF import ExtendedKalmanFilter
from Filter.singer_EKF import init_SingerModel
from Filter.KalmanNet import KalmanNet
from UI.analyze import analyze


torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)

if __name__ == "__main__":
    print_now_time()

    experiment = Exp()
    experiment.run(mode="train", dataset_name="train")

    #     if epoch_i % 2 == 0:
    #         # 计算当前的模型数量
    #         checkpoint_count = 0
    #         files = os.listdir("./")
    #         checkpoint_files = [x for x in files if "KalmanNet.pt" in x]
    #         checkpoint_files = sorted(
    #             checkpoint_files, key=lambda x: float(x.split("_")[0]))
    #         checkpoint_count = len(checkpoint_files)

    #         # 清除过多保存的模型
    #         if checkpoint_count > max_checkpoint_num:
    #             for i in range(0, checkpoint_count - max_checkpoint_num):
    #                 remove_checkpoint_name = "./"+checkpoint_files[-i-1]
    #                 if (os.path.exists(remove_checkpoint_name)):
    #                     os.remove(remove_checkpoint_name)
    #                 else:
    #                     print("要删除的文件不存在！")
    #         torch.save({
    #             'epoch': epoch_i,
    #             'model_state_dict': ekf.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'MSE': MSE_testset_mean[epoch_i],
    #             'MSE(dB)': MSE_dB_testset_mean[epoch_i],
    #         }, f"{MSE_dB_testset_mean[epoch_i]}_dB_epoch{epoch_i}_KalmanNet.pt")
    #     # elif MSE_testset_mean[epoch_i] >= 6*torch.min(MSE_testset_mean[0:epoch_i+1]):
    #     #     logger.warning(f"Begin Rewind!")
    #     #     files = os.listdir("./")
    #     #     checkpoint_files = [x for x in files if "KalmanNet.pt" in x]
    #     #     checkpoint_files = sorted(
    #     #         checkpoint_files, key=lambda x: float(x.split("_")[0]))
    #     #     checkpoint_rewind = torch.load(checkpoint_files[0])
    #     #     epoch_i = checkpoint_rewind["epoch"]
    #     #     ekf.load_state_dict(checkpoint_rewind["model_state_dict"])
    #     #     # optimizer.load_state_dict(checkpoint_rewind["optimizer_state_dict"])
    #     epoch_i += 1
    # # 绘制训练曲线
    # plt.switch_backend('agg')
    # plt.plot(MSE_dB_testset_mean.detach().cpu())
    # plt.savefig("./Result/training_curve.png")
    # torch.save(ekf.state_dict(), "testmodel.pkl")
    #analyze(ekf, x_ekf.detach(), x_true)
