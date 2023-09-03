import torch
from torch import nn
import numpy as np
class SingerMovementModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_matrix = torch.eye(2, dtype=torch.float32)
        self.tao = 5
        self.alpha = 0.1

    def forward(self, x):
        F1 = torch.hstack(
            (self.identity_matrix, self.tao * self.identity_matrix, (self.alpha * self.tao - 1 + np.exp(-self.alpha * self.tao)) / (self.alpha ** 2) * self.identity_matrix))
        # F1[2,:] = torch.tensor([0,0,1,0,0,0,0,0,0])
        F2 = torch.hstack((0 * self.identity_matrix, self.identity_matrix, (1 - np.exp(-self.alpha * self.tao)) / self.alpha * self.identity_matrix))
        # F2[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F3 = torch.hstack((0 * self.identity_matrix, 0 * self.identity_matrix, np.exp(-self.alpha * self.tao) * self.identity_matrix))
        # F3[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        F = torch.vstack((F1, F2, F3))
        return torch.matmul(F, x)

if __name__ == "__main__":
    model = SingerMovementModel()
    x = nn.Parameter(torch.rand([6,6]))
    y = model(x)
    target = torch.rand([6,6])
    loss_fn = nn.MSELoss()
    loss = loss_fn(y,target)
    loss.backward()