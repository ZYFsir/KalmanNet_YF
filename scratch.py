import torch

I = torch.tensor([[1,0],[0,1.5]], dtype=torch.float,device=torch.device("cuda:0"))
a = torch.tensor([[3,1,5],[2,3,6]], dtype=torch.float, device=torch.device("cuda:0"))
# b = I@a
