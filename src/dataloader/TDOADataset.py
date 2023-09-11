import torch
from torch.utils.data import Dataset, DataLoader

def file_filter(f):
    if f[-4:] in ['.mat']:
        return True
    else:
        return False


class TDOADataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.x = data["input"].transpose(1, 2).contiguous()
        self.y = data["target"].transpose(1, 2).contiguous()
        self.station = data["station"]
        self.h = data["h"].reshape([1,1])
        self.N, self.T,self.x_dim = self.x.shape
        self.y_dim = self.y.shape[-1]

    def __getitem__(self, item):
        return (self.x[item, :, :], self.y[item, :, :])

    def __len__(self):
        return self.N

