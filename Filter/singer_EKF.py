import torch
import numpy as np
from utils import logger, device, config

def init_SingerModel(station, dim, z):
    z = torch.tensor(z, dtype=torch.float, device=device)

    q = 1e4  # 模型噪声
    r = 1e-3  # 测量噪声

    I = torch.eye(2, dtype=torch.float, device=device)
    tao = 1
    alpha = 0.1

    def f(x):
        F1 = torch.hstack((I, tao * I, (alpha * tao - 1 + np.exp(-alpha * tao)) / (alpha ** 2) * I))
        # F1[2,:] = torch.tensor([0,0,1,0,0,0,0,0,0])
        F2 = torch.hstack((0 * I, I, (1 - np.exp(-alpha * tao)) / alpha * I))
        # F2[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F3 = torch.hstack((0 * I, 0 * I, np.exp(-alpha * tao) * I))
        # F3[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F = torch.vstack((F1, F2, F3))
        return torch.matmul(F, x).squeeze()

    def h(x):
        u = torch.hstack((x[0:2], z))
        r = torch.norm(u - station, dim=1)
        tdoa = torch.abs(r[1:] - r[0])
        return tdoa.squeeze()

    Q_true = (q ** 2) * torch.eye(n)
    R_true = (r ** 2) * torch.eye(m)

    T = 100
    T_test = 152
    return SystemModel(f, Q_true, h, R_true, T, T_test)