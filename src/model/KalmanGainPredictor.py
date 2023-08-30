import torch
import torch.nn as nn
from src.tools.neural_network_utils import create_fully_connected


def init_gru_parameters(gru_module):
    for name, param in gru_module.named_parameters():
        if 'weight_ih' in name or 'weight_hh' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

def init_linear_parameters(fc_module):
    nn.init.kaiming_uniform_(fc_module.weight, nonlinearity='relu')
    fc_module.bias.data.fill_(0)

class KalmanGainPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, in_mult, out_mult):
        super(KalmanGainPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_mult = in_mult
        self.out_mult = out_mult

        self.init_h_Q = None
        self.init_h_Sigma = None
        self.init_h_S = None
        self.initialize_network()
        self.initialize_parameters()

    def initialize_network(self):
        self.d_input_Q = self.output_dim * self.out_mult
        self.d_hidden_Q = self.output_dim ** 2
        self.init_h_Q = nn.Parameter(torch.zeros(self.d_hidden_Q))
        self.GRU_Q = nn.GRUCell(self.d_input_Q, self.d_hidden_Q)
        self.FC_Q = nn.Sequential(nn.Linear(self.d_hidden_Q, self.d_hidden_Q),
                                  nn.ReLU())
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.output_dim * self.out_mult
        self.d_hidden_Sigma = self.input_dim ** 2
        self.init_h_Sigma = nn.Parameter(torch.zeros(self.d_hidden_Sigma))
        self.GRU_Sigma = nn.GRUCell(self.d_input_Sigma, self.d_hidden_Sigma)
        self.FC_Sigma = nn.Sequential(nn.Linear(self.d_hidden_Sigma, self.d_hidden_Sigma),
                                      nn.ReLU())
        # GRU to track S
        self.d_input_S = self.output_dim ** 2 + 2 * self.input_dim * self.in_mult
        self.d_hidden_S = self.output_dim ** 2 + 2 * self.input_dim * self.in_mult
        self.init_h_S = nn.Parameter(torch.zeros(self.d_hidden_S))
        self.GRU_S = nn.GRUCell(self.d_input_S, self.d_hidden_S)
        self.FC_S = nn.Sequential(nn.Linear(self.d_input_S, self.d_hidden_S),
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
            hidden_in['Q'] = self.init_h_Q.unsqueeze(0).repeat(batch_size,1)
            hidden_in['Sigma'] = self.init_h_Sigma.unsqueeze(0).repeat(batch_size,1)
            hidden_in['S'] = self.init_h_S.unsqueeze(0).repeat(batch_size,1)
        hidden_out = {}
        obs_diff, obs_innovation_diff, fw_evol_diff, fw_update_diff = [
            torch.tensor(data).to(torch.float32).squeeze(2) for data in x]

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
