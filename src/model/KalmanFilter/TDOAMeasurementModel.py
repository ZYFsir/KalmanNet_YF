import torch
from torch import nn
class TDOAMeasurementModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, station, h):
        batch_size = x.shape[0]
        h_extended = h.unsqueeze(1).expand(batch_size, -1, -1)
        u = torch.cat((x[:,0:2,:], h_extended), dim=1)
        u = u.transpose(1,2)
        r = torch.norm(u - station.cuda(), dim=2, keepdim=True)
        tdoa = r[:,1:] - r[:,0:1]
        return tdoa

    def jacobian(self):
        # Compute and return the Jacobian matrix of the measurement prediction
        jacobian_matrix = ...  # Your Jacobian matrix calculation here
        return jacobian_matrix