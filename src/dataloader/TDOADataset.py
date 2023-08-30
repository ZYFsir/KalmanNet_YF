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
        self.x = data["input"]
        self.y = data["target"]
        self.station = data["station"]
        self.h = data["h"]
        self.N, self.T,self.x_dim = self.x.shape
        self.y_dim = self.y.shape[-1]

    def __getitem__(self, item):
        return (self.x[item, :, :], self.y[item, :, :], self.station, self.h)

    def __len__(self):
        return self.N

