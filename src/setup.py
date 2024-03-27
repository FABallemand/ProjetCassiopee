import os
import sys
import logging

import torch

def setup_python():
    # Check Python version
    logging.info(f"sys.version = {sys.version}")
    logging.info(f"os.getcwd() = {os.getcwd()}")


def setup_pytorch() -> torch.device:
    # Check PyTorch version
    logging.info(f"torch.__version__ = {torch.__version__}")

    # Enable faster runtime
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if torch.backends.cudnn.is_available():
        logging.info(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    # Set-up PyTorch device
    DEVICE = None
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        logging.info(f"torch.version.cuda = {torch.version.cuda}")
        torch.multiprocessing.set_start_method("spawn") # MAYBE NOT REQUIRED
    else:
        DEVICE = torch.device("cpu")
    # torch.set_default_device(DEVICE) # Requires newer PyTorch version
    logging.info(f"DEVICE = {DEVICE}")

    return DEVICE