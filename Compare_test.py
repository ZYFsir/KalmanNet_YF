import torch
import matplotlib.pyplot as plt

EKF_loss = torch.load("Result/test_loss/EKF_train_loss.pt")
KNet_loss = torch.load("Result/test_loss/KNet_train_loss.pt")

EKF_loss_trajs = torch.mean(EKF_loss,dim=[1,2])
KNet_loss_trajs = torch.mean(KNet_loss,dim=[1,2])

plt.plot(KNet_loss_trajs.cpu().numpy())
plt.plot(EKF_loss_trajs.cpu().numpy())
# plt.plot(b['MSE_per_batch'].cpu().numpy())
plt.legend(["KFNet", "EKF"])
plt.show()
