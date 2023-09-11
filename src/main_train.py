import torch
import os
from src.tools.torchSettings import print_now_time
from src.tools.Experiment import Experiment
from src.config.config import Config
from src.tools.TBPTT import TBPTT

torch.set_printoptions(precision=12)

if __name__ == "__main__":
    print_now_time()
    os.chdir('..')
    config = Config('src/config/exp01_kalmanNet.yaml')
    experiment = Experiment(config)
    dataset_name = 'train'
    experiment.train(dataset_name)
    print("Training Finished")

