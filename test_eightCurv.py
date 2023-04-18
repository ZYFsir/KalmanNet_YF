import numpy as np
import os, sys, argparse, torch, numpy

from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.torchSettings import get_torch_device, get_config, print_now_time
from Dataset.TDOADataset import MetaDataLoader
from utils import logger, device, config

from Filter.EKF import ExtendedKalmanFilter
from Filter.singer_EKF import init_SingerModel
from Filter.KalmanNet import KalmanNet
from UI.analyze import analyze
from UI.chart import plot_enlarge
torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    torch.set_printoptions(precision=12)
    print_now_time()
    dataloader = DataLoader(**(MetaDataLoader().dataloader_params))
    dataset_size = dataloader.sampler.data_source.datasets[0].N  # TODO: 多数据集情况时需要修改

    batch_size = config["dataloader_params"]["batch_size"]
    if dataset_size % batch_size != 0:
        logger.error(f"dataset size {dataset_size} 并非batch size{batch_size}的整数")
        exit()
    ## using KalmanNet
    ekf = KalmanNet()
    ekf.to(device)

    loss_fn = nn.MSELoss(reduction='mean')  # keep batch

    not_init_model = True
    MSE_epoch = []

    ## 载入最优模型checkpoint
    files = os.listdir("./")
    checkpoint_files = [x for x in files if "KalmanNet.pt" in x]
    checkpoint_files = sorted(checkpoint_files, key=lambda x: float(x.split("_")[0]))
    checkpoint_rewind = torch.load(checkpoint_files[0], map_location=device)
    # epoch_i = checkpoint_rewind["epoch"]
    ekf.load_state_dict(checkpoint_rewind["model_state_dict"])
    # optimizer.load_state_dict(checkpoint_rewind["optimizer_state_dict"])
    ekf.eval()

    MSE_testset_singledata = torch.empty([dataset_size // batch_size])
    MSE_dB_testset_singledata = torch.empty([dataset_size // batch_size])

    for data_i, data in enumerate(dataloader):
        if not_init_model:
            singer_model = init_SingerModel(data["station"], data["h"])
            not_init_model = False
            ekf.set_ssmodel(singer_model)

        (batch_size, T, n) = data["z"].shape
        ekf.batch_size = batch_size
        ekf.init_hidden()
        ekf.InitSequence()

        m = config["dim"]["state"]
        n = config["dim"]["measurement"]

        x_ekf = torch.empty([batch_size, T, m])

        x_true = data["x"].cuda()
        z = data["z"].cuda()
        for t in range(0, T):
            m1x_posterior = ekf.forward(z[:, t, :])
            x_ekf[:, t, :] = m1x_posterior.squeeze(2)

        ## training KalmanNet
        loss = loss_fn(x_true, x_ekf[:, :, 0:2])

        MSE_testset_singledata[data_i] = torch.mean(loss).item()
        MSE_dB_testset_singledata[data_i] = 10 * torch.log10(torch.mean(loss)).item()
        logger.info(f"{data_i + 1}/{dataset_size // batch_size} MSE:{MSE_testset_singledata[data_i]}")
        logger.info(f"{data_i + 1}/{dataset_size // batch_size} MSE(dB):{MSE_dB_testset_singledata[data_i]}")

        if data_i%500==0:
            plot_enlarge(x_true, x_ekf, data_i, MSE_dB_testset_singledata[data_i], "kalmanNet")

    MSE_testset_mean = torch.mean(MSE_testset_singledata).item()
    MSE_dB_testset_mean = torch.mean(MSE_dB_testset_singledata).item()
    torch.save({
        'MSE_testset_singledata': MSE_testset_singledata,
        'MSE_dB_testset_singledata': MSE_dB_testset_singledata,
        'MSE_testset_mean': MSE_testset_mean,
        'MSE_dB_testset_mean': MSE_dB_testset_mean
    }, "KFNet_error.pt")
    logger.info(f"MSE:{MSE_testset_mean}")
    logger.info(f"MSE(dB):{MSE_dB_testset_mean}")