import os
import sys
from subprocess import check_output
import logging

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

def setup_python():
    # Check Python version
    logging.info(f"sys.version = {sys.version}")
    logging.info(f"os.getcwd() = {os.getcwd()}")


def setup_pytorch(gpu=True) -> torch.device:
    # Check PyTorch version
    logging.info(f"torch.__version__ = {torch.__version__}")

    # Set-up PyTorch device
    DEVICE = None
    if gpu and torch.cuda.is_available():
        # Enable faster runtime
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        if torch.backends.cudnn.is_available():
            logging.info(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = True

        output = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        logging.info(f"{output.decode('utf-8')}")
        logging.info(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        logging.info(f"torch.version.cuda = {torch.version.cuda}")
        logging.info(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
        logging.info(f"torch.cuda.get_device_name(torch.cuda.current_device()) = {torch.cuda.get_device_name(torch.cuda.current_device())}")
        DEVICE = torch.device("cuda:0")
        torch.multiprocessing.set_start_method("spawn") # MAYBE NOT REQUIRED
        torch.cuda.empty_cache()

        # Force CUDA initialisation
        torch.cuda.empty_cache()
        s = 32
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=DEVICE), torch.zeros(s, s, s, s, device=DEVICE))
        torch.cuda.empty_cache()
    else:
        DEVICE = torch.device("cpu")

    # torch.set_default_device(DEVICE) # Requires newer PyTorch version
    logging.info(f"DEVICE = {DEVICE}")

    return DEVICE