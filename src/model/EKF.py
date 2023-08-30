import torch
from functorch import jacrev
from torch import vmap
from torch import autograd
import sys
import matplotlib.pyplot as plt


class ExtendedKalmanFilter(torch.nn.Module):
    """
    EKF中的SystemModel包含两项方程，而初始值、滤波中间值应当由滤波器进行保存。
    """
    # 先调用InitSequence进行x初始化
    # 再调用GenerateSequence进行滤波。测量值是一次性输入的
    #
    # 定义输入（测量值）为z，维度为m
    # 状态为x，维度为n

    def __init__(self):
        super().__init__()
        self.f = None  # 运动模型
        self.f_batch = None
        self.m = None  # 输入维度（测量值维度）
        self.Q = None  # 运动模型噪声

        # Has to be transformed because of EKF non-linearity
        self.h = None
        self.h_batch = None
        self.n = None  # 状态维度
        self.R = None  # 测量噪声

        self.m1x_prior_list = []
        self.m1x_posterior_list = []

        # self.KG_array = torch.zeros([batch_size, self.n, self.m])
        self.m1x_0 = None
        self.m1x_posterior = None
        self.m2x_0 = None
        self.inverse_batch = None
        # Full knowledge about the model or partial? (Should be made more elegant)
        self.fString = None
        self.hString = None

    def set_ssmodel(self, SystemModel, mode='full'):
        self.f = SystemModel.f  # 运动模型
        self.f_batch = vmap(self.f)
        self.m = SystemModel.state_dim  # 输入维度（测量值维度）
        self.Q = SystemModel.Q  # 运动模型噪声

        # Has to be transformed because of EKF non-linearity
        self.h = SystemModel.h
        self.h_batch = vmap(self.h)
        self.n = SystemModel.observation_dim  # 状态维度
        self.R = SystemModel.R  # 测量噪声

        self.m1x_prior_list = []
        self.m1x_posterior_list = []

        #self.KG_array = torch.zeros([batch_size, self.n, self.m])
        self.m1x_0 = SystemModel.m1x_0
        self.m1x_posterior = SystemModel.m1x_0
        self.m2x_0 = SystemModel.m2x_0
        self.inverse_batch = vmap(torch.linalg.inv)
        # Full knowledge about the model or partial? (Should be made more elegant)
        if (mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif (mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

    # Predict
    def Predict(self):
        # self.m1x_posterior上一次迭代完成的状态向量
        # Predict the 1-st moment of x
        self.m1x_prior = self.f_batch(self.m1x_posterior)
        # Compute the Jacobians，更新后的雅可比矩阵存到F和H中
        F = self.jacobianBatch(self.m1x_posterior, self.fString)
        H = self.jacobianBatch(self.m1x_prior, self.hString)
        self.UpdateJacobians(F, H)
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h_batch(self.m1x_prior)
        # Predict the 2-nd moment of y
        tmp = torch.bmm(self.H, self.m2x_prior)
        self.m2y = torch.bmm(tmp, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        # self.m2x_prior是状态x的协方差矩阵，H_T是测量矩阵的转置
        # self.m2y是测量值的协方差矩阵
        P12 = torch.bmm(self.m2x_prior, self.H_T)
        self.KG = torch.bmm(P12, self.inverse_batch(self.m2y))

        # batch_div = vmap(torch.div)
        # self.KG = batch_div(P12, self.m2y)
        # Save KalmanGain
        # self.KG_array[:,self.i,:,:] = self.KG
        # self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + \
            torch.bmm(self.KG, self.dy.unsqueeze(dim=2))

        # Compute the 2-nd posterior moment
        # self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        # self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)
        HP = torch.bmm(self.H, self.m2x_prior)
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, HP)

    def Update(self, y):
        self.Predict()      # 预测
        self.m1x_prior_list.append(self.m1x_prior[0, 0].cpu())
        self.KGain()        # 计算KG
        self.Innovation(y)  # 引入测量值
        self.Correct()      # 更新
        self.m1x_posterior_list.append(self.m1x_posterior[0, 0].cpu())
        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0.cuda()
        self.m2x_0 = m2x_0.cuda()

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F, -1, -2)
        self.H = H
        self.H_T = torch.transpose(H, -1, -2)

    def forward(self, z):
        # Pre allocate an array for predicted state and variance
        # 生成存储输入的空间
        (batch_size, T, n) = z.shape
        self.batch_size = batch_size
        self.x = torch.empty(size=[batch_size, T, self.m])
        self.sigma = torch.empty(size=[batch_size, T, self.m, self.m])
        self.KG_array = torch.empty((batch_size, T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation
        # squeeze用于维度压缩，将大小仅为1的维度全部删除，用于避免1x1x1当成维度1的这种bug

        self.m1x_0 = self.m1x_0.repeat((batch_size, 1, 1))
        self.m2x_0 = self.m2x_0.repeat((batch_size, 1, 1))
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0
        for t in range(0, T):
            zt = torch.squeeze(z[:, t, :])
            xt, sigmat = self.Update(zt)    # 新增输入，获得输出
            self.i += 1
            self.x[:, t, :] = torch.squeeze(xt)
            self.sigma[:, t, :, :] = torch.squeeze(sigmat)
        return self.x

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
        jac = vmap(jacrev(g))(x)
        jac_reshape = jac.reshape([self.batch_size, f_out, f_in])
        return jac_reshape

    def jaccsd(self, fun, x):
        x = x.squeeze()
        z = fun(x)
        n = x.size()[0]
        m = z.size()[0]
        A = torch.zeros([m, n])
        h = x*0 + 1e-5
        x1 = torch.zeros(n)
        for k in range(0, n):
            x1.copy_(x)
            x1[k] += h[k]
            A[:, k] = (fun(x1)-z)/h[k]
        return A.unsqueeze(0)

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
