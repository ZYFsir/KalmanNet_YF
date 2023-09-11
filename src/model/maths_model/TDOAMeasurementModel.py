import torch
from torch import nn
from torch.func import jacrev, vmap
class TDOAMeasurementModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.station = None
        self.h = None

    def forward(self, x, station, h):
        station = station.to(x.device)
        self.station = station
        self.h = h

        tdoa = vmap(self._forward)(x)
        return tdoa

    def _forward(self, x):
        # 由于需要借助vmap求jacobian，因此需要写一个非batch的_forward
        u = torch.cat((x[0:2, :], self.h), dim=0)
        r = torch.norm(u.T - self.station, dim=1, keepdim=True)
        tdoa = r[1:] - r[0:1]
        return tdoa

    def get_jacobian(self, x, station, h):
        station = station.to(x.device)
        self.station = station
        self.h = h

        jacobian_batch = vmap(jacrev(self._forward))
        return jacobian_batch(x)

if __name__ == "__main__":
    model = TDOAMeasurementModel()
    x = torch.rand([8, 6, 1])
    station = torch.rand([4,3])
    h = torch.rand([1,1])
    y = model(x, station, h)
    J = model.get_jacobian(x, station, h)
    print(J)