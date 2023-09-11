import torch
from torch import nn

class KalmanGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.hh = nn.GRU(1,2)
    def forward(self, F, H, Sigma_previous, Q, R):
        Sigma = F @ Sigma_previous @ F.T + Q
        measurement_predict_covariance = H @ Sigma @ H.T + R
        P12 = Sigma @ H.T
        KG = P12 @ torch.linalg.inv(measurement_predict_covariance)
        return KG