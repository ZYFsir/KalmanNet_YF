"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch
from functorch import jacrev, vmap
from filing_paths import path_model
from torch import autograd
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, path_model)

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")


class ExtendedKalmanFilter(torch.nn.Module):
    # 先调用InitSequence进行x初始化
    # 再调用GenerateSequence进行滤波。测量值是一次性输入的
    #
    # 定义输入（测量值）为z，维度为m
    # 状态为x，维度为n
    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f  # 运动模型
        self.f_batch = vmap(self.f)
        self.m = SystemModel.m  # 输入维度（测量值维度）

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q # 运动模型噪声

        self.h = SystemModel.h
        self.h_batch = vmap(self.h)
        self.n = SystemModel.n  # 状态维度

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R  # 测量噪声

        self.T = SystemModel.T  # 两种数据集的长度
        self.T_test = SystemModel.T_test

        self.m1x_prior_list = []
        self.m1x_posterior_list = []

        #self.KG_array = torch.zeros([batch_size, self.n, self.m])
        self.m1x_0 = None
        self.m2x_0 = None

        self.inverse_batch = vmap(torch.inverse)

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
        A = self.jacobianBatch(self.m1x_posterior, self.fString)
        B = self.jacobianBatch(self.m1x_prior, self.hString)
        self.UpdateJacobians(A, B)
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h_batch(self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        # self.m2x_prior是状态x的协方差矩阵，H_T是测量矩阵的转置
        # self.m2y是测量值的协方差矩阵
        self.KG = torch.bmm(self.m2x_prior, self.H_T)
        self.KG = torch.bmm(self.KG, self.inverse_batch(self.m2y))

        # Save KalmanGain
        self.KG_array[:,self.i,:,:] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy.unsqueeze(dim=2)).squeeze()

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)
        # self.m2x_posterior = torch.bmm(self.H, self.m2x_prior)
        # self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()      # 预测
        self.m1x_prior_list.append(self.m1x_prior[0,0].cpu())

        self.KGain()        # 计算KG
        self.Innovation(y)  # 引入测量值
        self.Correct()      # 更新
        self.m1x_posterior_list.append(self.m1x_posterior[0,0].cpu())

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        F = F.squeeze()
        self.F = F
        self.F_T = torch.transpose(F, -1, -2)
        self.H = H
        self.H_T = torch.transpose(H, -1, -2)
        # print(self.H,self.F,'\n')

    ### forward ###
    # 在进行任何
    def forward(self, y, m1x_0, m2x_0):
        # Pre allocate an array for predicted state and variance
        # 生成存储输入的空间
        (batch_size, T, n) = y.shape
        self.x = torch.empty(size=[batch_size, T, self.m])
        self.sigma = torch.empty(size=[batch_size, T, self.m, self.m])
        # Pre allocate KG array
        self.KG_array = torch.zeros((batch_size, T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation
        # squeeze用于维度压缩，将大小仅为1的维度全部删除，用于避免1x1x1当成维度1的这种bug
        self.m1x_0 = m1x_0.repeat((batch_size, 1, 1))
        self.m2x_0 = m2x_0.repeat((batch_size, 1, 1))

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0



        for t in range(0, T):
            yt = torch.squeeze(y[:, t, :])
            xt, sigmat = self.Update(yt)    # 新增输入，获得输出
            self.x[:, t, :] = torch.squeeze(xt)
            self.sigma[:, t, :, :] = torch.squeeze(sigmat)
        plt.plot(self.m1x_prior_list)
        plt.plot(self.m1x_posterior_list)



    def jacobianBatch(self, x, a):
        if (a == 'ObsAcc'):
            g = self.h
        elif (a == 'ModAcc'):
            g = self.f
        elif (a == 'ObsInacc'):
            g = self.hInacc
        elif (a == 'ModInacc'):
            g = self.fInacc
        return vmap(jacrev(g))(x)

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