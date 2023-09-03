import torch
from torch import nn
class TDOAMeasurementModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, station, h):
        station = station.to(x.device)
        batch_size = x.shape[0]
        h_extended = h.unsqueeze(1).expand(batch_size, -1, -1)
        u = torch.cat((x[:,0:2,:], h_extended), dim=1)
        u = u.transpose(1,2)
        r = torch.norm(u - station, dim=2, keepdim=True)
        tdoa = r[:,1:] - r[:,0:1]
        return tdoa

    def jacobian(self):
        # Compute and return the Jacobian matrix of the measurement prediction
        jacobian_matrix = ...  # Your Jacobian matrix calculation here
        return jacobian_matrix

if __name__ == "__main__":
    model = TDOAMeasurementModel()
    x = nn.Parameter(torch.rand([8, 6, 1]))
    station = torch.rand([8,4,3])
    h = torch.rand([8, 1])
    y = model(x, station, h)
    target = torch.rand([8,1,1])
    loss_fn = nn.MSELoss()
    loss = loss_fn(y[:,0:1,:],target)
    loss.backward()