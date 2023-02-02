import torch
from functorch import jacrev, vmap
from filing_paths import path_model
from torch import autograd, nn
import torch.nn.functional as func
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from Filter import ssmodel
from utils import logger, device, config

in_mult = config["in_mult"]
out_mult = config["out_mult"]

class KalmanNet(torch.nn.Module):
    """
    EKF中的SystemModel包含两项方程，而初始值、滤波中间值应当由滤波器进行保存。
    """
    # 先调用InitSequence进行x初始化
    # 再调用GenerateSequence进行滤波。测量值是一次性输入的
    #
    # 定义输入（测量值）为z，维度为m
    # 状态为x，维度为n
    def __init__(self, mode='full'):
        self.mode = mode
        super().__init__()
        # 设置维度
        self.m = config["dim"]["state"]
        self.n = config["dim"]["measurement"]
        # 初始化NN网络
        self.InitKGainNet()

    def set_ssmodel(self, model):
        self.model = model
        self.f = model.f  # 运动模型
        self.f_batch = vmap(self.f)
        self.m = model.m  # 输入维度（测量值维度）
        self.Q = model.Q  # 运动模型噪声

        # Has to be transformed because of EKF non-linearity
        self.h = model.h
        self.h_batch = vmap(self.h)
        self.n = model.n  # 状态维度
        self.R = model.R  # 测量噪声
        self.m1x_0 = model.m1x_0
        self.m1x_posterior = model.m1x_0
        self.m2x_0 = model.m2x_0
        self.prior_Q = model.prior_Q
        self.prior_Sigma = model.prior_Sigma
        self.prior_S = model.prior_S

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F, -1, -2)
        self.H = H
        self.H_T = torch.transpose(H, -1, -2)

    def forward(self, z):
        # Pre allocate an array for predicted state and variance
        batch_size = z.shape[0]
        self.batch_size = batch_size
        # squeeze用于维度压缩，将大小仅为1的维度全部删除，用于避免1x1x1当成维度1的这种bug

        self.m1x_0 = self.m1x_0.repeat((batch_size, 1, 1))
        self.m2x_0 = self.m2x_0.repeat((batch_size, 1, 1))
        self.m1x_posterior = self.m1x_0
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h_batch(self.m1x_posterior)

        self.x = self.Update(z)    # 新增输入，获得输出
        return self.x

    def Update(self, y):
        self.Predict()      # 预测
        self.KGain(y)        # 计算KG
        self.Innovation(y)  # 引入测量值
        self.Correct(y)      # 更新
        return self.m1x_posterior

    # Predict
    def Predict(self):
        self.m1x_prior = self.f_batch(self.m1x_posterior)
        self.m1y = self.h_batch(self.m1x_prior)

    # Compute the Kalman Gain
    def KGain(self, y):
        # 先创建网络输入
        obs_diff = y - self.y_previous
        obs_innov_diff = y - self.m1y
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous

        # 网络输入归一化
        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        tmp_KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KG = torch.reshape(tmp_KG, [self.batch_size, self.m, self.n])
        # batch_div = vmap(torch.div)
        # self.KG = batch_div(P12, self.m2y)
        # Save KalmanGain
        # self.KG_array[:,self.i,:,:] = self.KG
        # self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self, y):
        INOV = torch.bmm(self.KG, self.dy.unsqueeze(2))
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        # self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

    def jacobianBatch(self, x, a):
        if (a == 'ObsAcc'):
            g = self.h
            f_out = self.n
            f_in = self.m
        elif (a == 'ModAcc'):
            g = self.f
            f_out = self.m
            f_in = self.m
        elif (a == 'ObsInacc'):
            f_out = self.n
            f_in = self.m
        elif (a == 'ModInacc'):
            g = self.fInacc
            f_out = self.m
            f_in = self.m
        jac=vmap(jacrev(g))(x)
        jac_reshape = jac.reshape([self.batch_size, f_out, f_in])
        return jac_reshape

    # def jaccsd(self, fun, x):
    #     x = x.squeeze()
    #     z = fun(x)
    #     n = x.size()[0]
    #     m = z.size()[0]
    #     A = torch.zeros([m,n])
    #     h = x*0 + 1e-5
    #     x1 = torch.zeros(n)
    #     for k in range(0,n):
    #         x1.copy_(x)
    #         x1[k]+=h[k]
    #         A[:,k] = (fun(x1)-z)/h[k]
    #     return A.unsqueeze(0)

    def getJacobian(self, x, a):
        # if(x.size()[1] == 1):
        #     y = torch.reshape((x.T),[x.size()[0]])
        try:
            if (x.size()[1] == 1):
                y = torch.reshape((x.T), [x.size()[0]])
        except:
            y = torch.reshape((x.T), [x.size()[0]])

        if (a == 'ObsAcc'):
            g = self.h
        elif (a == 'ModAcc'):
            g = self.f
        elif (a == 'ObsInacc'):
            g = self.hInacc
        elif (a == 'ModInacc'):
            g = self.fInacc

        Jac = autograd.functional.jacobian(g, y)
        Jac = Jac.view(-1, self.m)
        return Jac



    def InitKGainNet(self):
        self.seq_len_input = 1
        self.batch_size = 1
        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.batch_size, self.d_hidden_Q).to(device, non_blocking=True)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.batch_size, self.d_hidden_Sigma).to(device, non_blocking=True)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.batch_size, self.d_hidden_S).to(device, non_blocking=True)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU())

        """
        # Fully connected 8
        self.d_input_FC8 = self.d_hidden_Q
        self.d_output_FC8 = self.d_hidden_Q
        self.d_hidden_FC8 = self.d_hidden_Q * Q_Sigma_mult
        self.FC8 = nn.Sequential(
                nn.Linear(self.d_input_FC8, self.d_hidden_FC8),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC8, self.d_output_FC8))
        """

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        # obs_diff = expand_dim(obs_diff)
        # obs_innov_diff = expand_dim(obs_innov_diff)
        # fw_evol_diff = expand_dim(fw_evol_diff)
        # fw_update_diff = expand_dim(fw_update_diff)

        # obs_diff对应yt， obs_innov_diff对应~yt， fw_evol_diff对应^xt, fw_update_diff对应~xt
        obs_diff = torch.reshape(obs_diff, [self.batch_size, self.n])
        obs_innov_diff = torch.reshape(obs_innov_diff, [self.batch_size, self.n])
        fw_evol_diff = torch.reshape(fw_evol_diff, [self.batch_size, self.m])
        fw_update_diff = torch.reshape(fw_update_diff, [self.batch_size, self.m])

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 1)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 1)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 1)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 1)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 1)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 1)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    def init_SingerModel(self, station, height):
        m = config["dim"]["state"]
        n = config["dim"]["measurement"]
        station = station.cuda()
        height = height.cuda()

        r = config["Observation model"]["r"]  # 测量噪声

        I = torch.eye(2, dtype=torch.double, device=device)
        alpha = 0.1
        tao = 1
        sigma_a = 100
        R_corr = 0.5
        x_p = 1e6

        def f(x):
            F1 = torch.hstack((I, tao * I, (alpha * tao - 1 + np.exp(-alpha * tao)) / (alpha ** 2) * I))
            # F1[2,:] = torch.tensor([0,0,1,0,0,0,0,0,0])
            F2 = torch.hstack((0 * I, I, (1 - np.exp(-alpha * tao)) / alpha * I))
            # F2[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
            F3 = torch.hstack((0 * I, 0 * I, np.exp(-alpha * tao) * I))
            # F3[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
            F = torch.vstack((F1, F2, F3))
            return torch.matmul(F, x)

        def h(x):
            u = torch.hstack((x[0:2, 0], height))
            r = torch.norm(u - station, dim=2)
            tdoa = r[0, 1:] - r[0, 0]
            return tdoa

        alpha_tao = 0.05
        q11 = (sigma_a ** 2) / (alpha ** 4) * (1 - exp(
            -2 * alpha_tao) + 2 * alpha_tao + 2 / 3 * alpha_tao ** 3 - 2 * alpha_tao ** 2 - 4 * alpha_tao * exp(
            -alpha_tao))
        q12 = (sigma_a ** 2) / (alpha ** 3) * (exp(-2 * alpha_tao) + 1 - 2 * exp(-alpha_tao) + 2 * alpha_tao * exp(
            -alpha_tao) - 2 * alpha_tao + alpha_tao ** 2)
        q13 = (sigma_a ** 2) / (alpha ** 2) * (1 - exp(-2 * alpha_tao) - 2 * alpha_tao * exp(-alpha_tao))
        q22 = (sigma_a ** 2) / (alpha ** 2) * (4 * exp(-alpha_tao) - 3 - exp(-2 * alpha_tao) + 2 * alpha_tao)
        q23 = (sigma_a ** 2) / (alpha) * (exp(-2 * alpha_tao) + 1 - 2 * exp(-alpha_tao))
        q33 = sigma_a ** 2 * (1 - exp(-2 * alpha_tao))
        Q1 = torch.hstack((q11 * I, q12 * I, q13 * I))
        Q2 = torch.hstack((q12 * I, q22 * I, q23 * I))
        Q3 = torch.hstack((q13 * I, q23 * I, q33 * I))
        Q_true = torch.vstack((Q1, Q2, Q3))

        R_true = R_corr * (r ** 2) * (torch.ones(n) - torch.eye(n) + torch.eye(n) / R_corr)
        model = ssmodel(f, Q_true, h, R_true)

        model.InitSequence(torch.tensor([[1e4, 1e5, 0, 0, 0, 0]]).T, x_p * torch.ones([m, m]))
        return model


    def init_KalmanNet(self):
        self.f = self.model.f  # 运动模型
        self.f_batch = vmap(self.f)
        self.m = self.model.m  # 输入维度（测量值维度）
        self.Q = self.model.Q  # 运动模型噪声

        # Has to be transformed because of EKF non-linearity
        self.h = self.model.h
        self.h_batch = vmap(self.h)
        self.n = self.model.n  # 状态维度
        self.R = self.model.R  # 测量噪声

        self.m1x_prior_list = []
        self.m1x_posterior_list = []

        # self.KG_array = torch.zeros([batch_size, self.n, self.m])
        self.m1x_0 = self.model.m1x_0
        self.m1x_posterior = self.model.m1x_0
        self.m2x_0 = self.model.m2x_0
        self.inverse_batch = vmap(torch.linalg.inv)
        # Full knowledge about the model or partial? (Should be made more elegant)
        if (self.mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif (self.mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0,:] = self.prior_Q.flatten()