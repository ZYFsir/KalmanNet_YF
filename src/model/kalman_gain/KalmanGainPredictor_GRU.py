import torch
import torch.nn as nn
from tools.neural_network_utils import create_fully_connected
from tools.utils import state_detach
from numpy import exp


def init_gru_parameters(gru_module):
    for name, param in gru_module.named_parameters():
        if 'weight_ih' in name or 'weight_hh' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

def init_linear_parameters(fc_module):
    nn.init.kaiming_uniform_(fc_module.weight, nonlinearity='relu')
    fc_module.bias.data.fill_(0)

class KalmanGainPredictor_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(KalmanGainPredictor_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.init_h_Q = None
        self.init_h_Sigma = None
        self.init_h_S = None

        self.initialize()

    def initialize(self):
        # 定义各运算层
        self.initialize_network()
        # 初始化模型协方差矩阵
        self.initialize_covariance()
        # 初始化各层参数
        self.initialize_parameters()

    def initialize_covariance(self):
        alpha = 0.1
        tao = 5
        sigma_a = 10
        r = 10
        R_corr = 0.5
        x_p = 1e3
        alpha_tao = alpha * tao
        I = torch.eye(2)

        q11 = (sigma_a ** 2) / (alpha ** 4) * (1 - exp(
            -2 * alpha_tao) + 2 * alpha_tao + 2 / 3 * alpha_tao ** 3 - 2 * alpha_tao ** 2 - 4 * alpha_tao * exp(
            -alpha_tao))
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
        Q = Q_true.reshape(self.d_hidden_Q)

        Sigma_true = torch.eye(self.output_dim)
        Sigma = Sigma_true.reshape(self.d_hidden_Sigma)

        R_true = R_corr * (r ** 2) * (torch.ones(self.input_dim) -
                                      torch.eye(self.input_dim) + torch.eye(self.input_dim) / R_corr)
        S = R_true.reshape(self.d_hidden_S)
        self.init_h_Q = nn.Parameter(Q)
        self.init_h_Sigma = nn.Parameter(torch.zeros(self.d_hidden_Sigma))
        self.init_h_S = nn.Parameter(torch.zeros(self.d_hidden_S))

    def initialize_network(self):
        self.d_input_Q = self.output_dim * self.out_mult
        self.d_hidden_Q = self.output_dim ** 2
        self.GRU_Q = nn.GRUCell(self.d_input_Q, self.d_hidden_Q)
        self.FC_Q = nn.Sequential(nn.Linear(self.d_hidden_Q, self.d_hidden_Q),
                                  nn.ReLU())
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.output_dim * self.out_mult
        self.d_hidden_Sigma = self.output_dim ** 2
        self.GRU_Sigma = nn.GRUCell(self.d_input_Sigma, self.d_hidden_Sigma)
        self.FC_Sigma = nn.Sequential(nn.Linear(self.d_hidden_Sigma, self.d_hidden_Sigma),
                                      nn.ReLU())
        # GRU to track S
        self.d_input_S = self.output_dim ** 2 + 2 * self.input_dim * self.in_mult
        self.d_hidden_S = self.input_dim ** 2
        self.GRU_S = nn.GRUCell(self.d_input_S, self.d_hidden_S)
        self.FC_S = nn.Sequential(nn.Linear(self.d_hidden_S, self.d_hidden_S),
                                  nn.ReLU())
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.output_dim ** 2
        self.FC1 = create_fully_connected(self.d_input_FC1, self.d_output_FC1, 0)
        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.output_dim * self.input_dim
        self.d_hidden_FC2 = self.d_input_FC2 * self.out_mult
        self.FC2 = create_fully_connected(self.d_input_FC2, self.d_output_FC2, 1, self.d_hidden_FC2)
        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.output_dim ** 2
        self.FC3 = create_fully_connected(self.d_input_FC3, self.d_output_FC3, 0)
        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = create_fully_connected(self.d_input_FC4, self.d_output_FC4, 0)
        # Fully connected 5
        self.d_input_FC5 = self.output_dim
        self.d_output_FC5 = self.output_dim * self.out_mult
        self.FC5 = create_fully_connected(self.d_input_FC5, self.d_output_FC5, 0)
        # Fully connected 6
        self.d_input_FC6 = self.output_dim
        self.d_output_FC6 = self.output_dim * self.out_mult
        self.FC6 = create_fully_connected(self.d_input_FC6, self.d_output_FC6, 0)
        # Fully connected 7
        self.d_input_FC7 = 2 * self.input_dim
        self.d_output_FC7 = 2 * self.input_dim * self.in_mult
        self.FC7 = create_fully_connected(self.d_input_FC7, self.d_output_FC7, 0)

    def initialize_parameters(self):
        for module in self.children():
            if isinstance(module, nn.GRUCell):
                init_gru_parameters(module)
            if isinstance(module, nn.Linear):
                init_linear_parameters(module)
            # Add conditions for other module types if needed

    def forward(self, x, hidden_in):
        if hidden_in == None:
            batch_size = x[0].shape[0]
            hidden_in = {}
            hidden_in['Q'] = self.init_h_Q.unsqueeze(0).repeat(batch_size,1).data
            hidden_in['Sigma'] = self.init_h_Sigma.unsqueeze(0).repeat(batch_size,1).data
            hidden_in['S'] = self.init_h_S.unsqueeze(0).repeat(batch_size,1).data
        hidden_out = {}
        # obs_diff, obs_innovation_diff, fw_evol_diff, fw_update_diff = [
        #     torch.tensor(data).to(torch.float32).squeeze(2) for data in x]
        obs_diff, obs_innovation_diff, fw_evol_diff, fw_update_diff = [
            data.squeeze(2) for data in x]
        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        hidden_out["Q"] = self.GRU_Q(in_Q, hidden_in["Q"])
        output_Q = self.FC_Q(hidden_out["Q"])

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((output_Q, out_FC6), dim=1)
        hidden_out["Sigma"] = self.GRU_Sigma(in_Sigma, hidden_in["Sigma"])
        output_Sigma = self.FC_Sigma(hidden_out["Sigma"])

        # FC 1
        in_FC1 = output_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innovation_diff), 1)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 1)
        hidden_out["S"] = self.GRU_S(in_S, hidden_in["S"])
        output_S = self.FC_S(hidden_out["S"])

        # FC 2
        in_FC2 = torch.cat((output_Sigma, output_S), 1)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((output_S, out_FC2), 1)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((output_Sigma, out_FC3), 1)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        hidden_out["Sigma"] = out_FC4
        return out_FC2, hidden_out

if __name__ == "__main__":
    model = KalmanGainPredictor_GRU(3,6,20,60)
    x = [torch.rand([4, 3, 1]), torch.rand([4, 3, 1]),torch.rand([4, 6, 1]),torch.rand([4, 6, 1])]
    hidden = None
    y, hidden_out = model(x, hidden)
    target = torch.rand([4,18])
    loss_fn = nn.MSELoss()
    loss = loss_fn(y,target)
    loss.backward()

    z, hidden_out_2 = model(x, state_detach(hidden_out))
    loss_2 = loss_fn(z, target)
    loss_2.backward()

    print(hidden_out.grad)