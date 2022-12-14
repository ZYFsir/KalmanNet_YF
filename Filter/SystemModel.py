import torch
import numpy as np
from numpy import exp
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
from utils import logger, device, config
class SystemModel:
    """
    希望系统模型应当包括运动模型、测量模型
    """
    def __init__(self, f, Q, h, R, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.Q = Q
        self.m = self.Q.size()[0]   # 状态维度，即输出维度

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.R = R
        self.n = self.R.size()[0]   # 观测维度，即输入维度

        ################
        ### Sequence ###
        ################
        self.m1x_0 = None
        self.x_prev = None
        self.m2x_0 = None

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            xt = self.f(self.x_prev)

            # Process Noise
            mean = torch.zeros(self.m).cpu()
            eq = np.random.multivariate_normal(mean, Q_gen.cpu(), 1)
            eq = torch.transpose(torch.tensor(eq), 0, 1)
            eq = eq.type(torch.float)

            # Additive Process Noise
            xt = xt.add(eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)

            # Observation Noise
            mean = torch.zeros(self.n)
            er = np.random.multivariate_normal(mean.cpu(), R_gen.cpu(), 1)
            er = torch.transpose(torch.tensor(er), 0, 1)

            # Additive Observation Noise
            yt = yt.add(er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x


    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = np.transpose(Aq) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = np.transpose(Ar) * Ar

        return [Q_gen, R_gen]

def init_SingerModel(station, height):
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
        u = torch.hstack((x[0:2,0], height))
        r = torch.norm(u - station, dim=2)
        tdoa = r[0,1:] - r[0,0]
        return tdoa

    alpha_tao = 0.05
    q11 = (sigma_a ** 2) / (alpha ** 4) * (1 - exp(
        -2 * alpha_tao) + 2 * alpha_tao + 2 / 3 * alpha_tao ** 3 - 2 * alpha_tao ** 2 - 4 * alpha_tao * exp(-alpha_tao))
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

    R_true =  R_corr * (r ** 2) * (torch.ones(n) -torch.eye(n)+ torch.eye(n) /R_corr)
    model = SystemModel(f, Q_true, h, R_true)

    model.InitSequence(torch.tensor([[1e4,1e5,0,0,0,0]]).T, x_p*torch.ones([m,m]))
    return model