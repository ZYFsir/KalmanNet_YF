import torch
import numpy as np
from numpy import exp
from Filter.ssmodel import SystemModel


def init_SingerModel(station, z, m, n, r, device):
    station = station.cuda()
    if z.shape != 1:
        z = z[0]
    z = z.cuda()

    I = torch.eye(2, dtype=torch.float32, device=device)
    alpha = 0.1
    tao = 1
    sigma_a = 100
    R_corr = 0.5
    x_p = 1e6

    def f(x):
        F1 = torch.hstack(
            (I, tao * I, (alpha * tao - 1 + np.exp(-alpha * tao)) / (alpha ** 2) * I))
        # F1[2,:] = torch.tensor([0,0,1,0,0,0,0,0,0])
        F2 = torch.hstack((0 * I, I, (1 - np.exp(-alpha * tao)) / alpha * I))
        # F2[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F3 = torch.hstack((0 * I, 0 * I, np.exp(-alpha * tao) * I))
        # F3[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F = torch.vstack((F1, F2, F3))
        return torch.matmul(F, x)

    def h(x):
        u = torch.hstack((x[0:2, 0], z))     # TODO: 注意这里假定了所有数据中z均一样
        r = torch.norm(u - station, dim=2)
        tdoa = r[0, 1:] - r[0, 0]
        return tdoa

    alpha_tao = 0.05
    q11 = (sigma_a ** 2) / (alpha ** 4) * (1 - exp(
        -2 * alpha_tao) + 2 * alpha_tao + 2 / 3 * alpha_tao ** 3 - 2 * alpha_tao ** 2 - 4 * alpha_tao * exp(-alpha_tao))
    q12 = (sigma_a ** 2) / (alpha ** 3) * (exp(-2 * alpha_tao) + 1 - 2 * exp(-alpha_tao) + 2 * alpha_tao * exp(
        -alpha_tao) - 2 * alpha_tao + alpha_tao ** 2)
    q13 = (sigma_a ** 2) / (alpha ** 2) * \
        (1 - exp(-2 * alpha_tao) - 2 * alpha_tao * exp(-alpha_tao))
    q22 = (sigma_a ** 2) / (alpha ** 2) * \
        (4 * exp(-alpha_tao) - 3 - exp(-2 * alpha_tao) + 2 * alpha_tao)
    q23 = (sigma_a ** 2) / (alpha) * \
        (exp(-2 * alpha_tao) + 1 - 2 * exp(-alpha_tao))
    q33 = sigma_a ** 2 * (1 - exp(-2 * alpha_tao))
    Q1 = torch.hstack((q11 * I, q12 * I, q13 * I))
    Q2 = torch.hstack((q12 * I, q22 * I, q23 * I))
    Q3 = torch.hstack((q13 * I, q23 * I, q33 * I))
    Q_true = torch.vstack((Q1, Q2, Q3))

    R_true = R_corr * (r ** 2) * (torch.ones(n) -
                                  torch.eye(n) + torch.eye(n) / R_corr)
    model = SystemModel(f, Q_true, h, R_true)

    model.InitSequence(torch.tensor(
        [[1e4, 1e5, 0, 0, 0, 0]]).T, x_p * torch.ones([m, m]))
    return model
