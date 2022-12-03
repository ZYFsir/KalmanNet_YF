import torch
import json
from utils import logger


def get_torch_device():
    if torch.cuda.is_available():
       device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
       torch.set_default_tensor_type('torch.cuda.FloatTensor')
       logger.info("Running on the GPU")
    else:
       device = torch.device("cpu")
       logger.info("Running on the CPU")
    return device

def print_now_time():
    from datetime import datetime
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    logger.info(f"Current Time ={strTime}")

def get_config(path = "Config/config.json"):
    with open(path, 'r') as f:
        content = f.read()
        config = json.loads(content)
    return config

