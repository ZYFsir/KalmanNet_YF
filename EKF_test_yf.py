import torch.nn as nn
import torch
import time
from EKF_yf import ExtendedKalmanFilter
import matplotlib.pyplot as plt

def EKFTest(SysModel, dataset, N_T,batch_size, modelKnowledge = 'full', allStates=True):
    # LOSS
    loss_fn_each = nn.MSELoss(reduction='none')

    dataset_size = dataset.sampler.num_samples
    sampler_size = dataset.sampler.num_samples // batch_size
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty([batch_size, sampler_size])

    #KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty((sampler_size, batch_size, N_T, SysModel.m))
    start = time.time()
    for (j, data) in enumerate(dataset):
        EKF.forward(data['input'], SysModel.m1x_0, SysModel.m2x_0)
        plt.plot(data['target'][0,:,0].cpu())
        plt.show()
        if(allStates):

            loss_each = torch.mean(loss_fn_each(EKF.x[:,:,0:3], data['target']), dim=[1,2])
            MSE_EKF_linear_arr[:,j] = loss_each
        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr[j] = loss_fn_each(EKF.x[loc,:], data['target']).item()
        #KG_array = torch.add(EKF.KG_array, KG_array)
        EKF_out[j,:,:,:] = EKF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    #KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)
    
    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_out]



