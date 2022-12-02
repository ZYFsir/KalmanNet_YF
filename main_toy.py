import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from KalmanNet_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from datetime import datetime

from EKF_test import EKFTest
from UKF_test import UKFTest
from PF_test import PFTest

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n
from model import f, h, fInacc

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

   

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'


r2 = torch.tensor([16, 4, 1, 0.01, 1e-4])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
qopt = torch.sqrt(q2)
# qopt = torch.tensor([0.2, 4, 1, 0.1, 0.01])

for index in range(0,len(r2)):
   ####################
   ### Design Model ###
   ####################
   
   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   Q_true = (q2[index]) * torch.eye(m)
   R_true = (r2[index]) * torch.eye(n)
   sys_model = SystemModel(f, Q_true, h, R_true, T, T_test)
   sys_model.InitSequence(m1x_0, m2x_0)

   # Mismatched model
   sys_model_partial = SystemModel(fInacc, Q_true, h, R_true, T, T_test)
   sys_model_partial.InitSequence(m1x_0, m2x_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataFolderName = 'Simulations/Toy_problems' + '/'
   dataFileName = 'T100.pt'
   print("Start Data Gen")
   DataGen(sys_model, dataFolderName + dataFileName, T, T_test,randomInit=False)
   print("Data Load")
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())



   ################################
   ### Evaluate EKF, UKF and PF ###
   ################################
   
#    print("Searched optimal 1/q2 [dB]: ", 10 * torch.log10(1/qopt[index]**2))
#    Q_search = (qopt[index]**2) * torch.eye(m)
#    sys_model = SystemModel(f, Q_search, h, R_true, T, T_test, m, n,"Toy")
#    sys_model.InitSequence(m1x_0, m2x_0)

#    sys_model_partial = SystemModel(fInacc, Q_search, h, R_true, T, T_test, m, n,"Toy")
#    sys_model_partial.InitSequence(m1x_0, m2x_0)
#    print("Evaluate Kalman Filter True")
#    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
#    print("Evaluate Kalman Filter Partial")
#    [MSE_KF_linear_arr_partial, MSE_KF_linear_avg_partial, MSE_KF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partial, test_input, test_target)

#    print("Evaluate UKF True")
#    [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model, test_input, test_target)
#    print("Evaluate UKF Partial")
#    [MSE_UKF_linear_arr_partial, MSE_UKF_linear_avg_partial, MSE_UKF_dB_avg_partial, UKF_out_partial] = UKFTest(sys_model_partial, test_input, test_target)
  
#    print("Evaluate PF True")
#    [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model, test_input, test_target)
#    print("Evaluate PF Partial")
#    [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial] = PFTest(sys_model_partial, test_input, test_target)


   # DatafolderName = 'Data' + '/'
   # DataResultName = '10x10_Ttest1000' 
   # torch.save({
   #             'MSE_KF_linear_arr': MSE_KF_linear_arr,
   #             'MSE_KF_dB_avg': MSE_KF_dB_avg,
   #             }, DatafolderName+DataResultName)

   ##################
   ###  KalmanNet ###
   ##################
   print("KNet with full model info")
   modelFolder = 'KNet' + '/'
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   KNet_Pipeline.setssModel(sys_model)
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1e-3, weightDecay=1e-4)

   # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

   KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
   [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
   KNet_Pipeline.save()

   # KNet with model mismatch
   ## Build Neural Network
   print("KNet with partial model info")
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model_partial)
   # Model = torch.load('KNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
   ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet_partial")
   KNet_Pipeline.setssModel(sys_model_partial)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
   KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
   ## Test Neural Network
   [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
   KNet_Pipeline.save()