import sys

import torch

def setup_python():
    # Check Python version
    print(f"sys.version = {sys.version}")


def setup_pytorch():
    # Check PyTorch version
    print(f"torch.__version__ = {torch.__version__}")

    # Enable faster runtime
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if torch.backends.cudnn.is_available():
        print(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
        if not torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    # Set-up PyTorch device
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE = {DEVICE}")
    # torch.set_default_device(DEVICE) # Requires newer PyTorch version
    if torch.cuda.is_available():
        print(f"torch.version.cuda = {torch.version.cuda}")
        torch.multiprocessing.set_start_method("spawn") # MAYBE NOT REQUIRED