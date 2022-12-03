# Logger创建
from UI.logger import init_logger, get_logger
init_logger()
logger = get_logger()

# Config读取
from utils.torchSettings import get_torch_device, get_config
device = get_torch_device()
config = get_config()


