import torch

def get_torch_device():
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # print("Running on the GPU")
    else:
        device = torch.device("cpu")
        # print("Running on the CPU")
    return device


def print_now_time():
    from datetime import datetime
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print(f"Current Time ={strTime}")

