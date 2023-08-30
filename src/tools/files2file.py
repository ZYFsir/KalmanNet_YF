# 本代码用于将数据集中的各个文件整合到单个文件中储存，并观察能否提高运行速度
import os
import scipy.io as io
import numpy as np
import torch
# 设置
path = "../../data/raw/SNR5_N161/train_3800"
print(f"matlab数据集路径：{path}")
# 读取所有数据
def file_filter(f):
    if f[-4:] in ['.mat']:
        return True
    else:
        return False

files = os.listdir(path)
files = list(filter(file_filter, files))
N = len(files)
print(f"共计{N}条数据")
for idx in range(0, N):
    data = io.loadmat(os.path.join(path, files[idx]), mat_dtype=False)
    if idx == 0:
        T, m = data['test_tdoa'].shape
        input = torch.zeros((N, T, m))
        target = torch.zeros((N, T, 2))
    input[idx, :, :] = torch.Tensor(data['test_tdoa'])
    target[idx, :, :] = torch.Tensor(data["test_data"][:,0:2])
station = torch.Tensor(data["test_station"])
station = torch.reshape(station,[4,3])
h = torch.FloatTensor([data["test_data"][0,-1]])

target_path = "../../data/processed/train_3800.pt"
torch.save({"input":input,"target":target, "station":station, "h":h}
           ,target_path)
print(f"处理完毕，已存储到{target_path}")