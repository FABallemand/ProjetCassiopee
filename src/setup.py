import sys

import torch

def setup_python():
    # Check Python version
    print(f"sys.version = {sys.version}")


def setup_pytorch() -> torch.device:
    # Check PyTorch version
    print(f"torch.__version__ = {torch.__version__}")

    # Enable faster runtime
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if torch.backends.cudnn.is_available():
        print(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    # Set-up PyTorch device
    DEVICE = None
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        print(f"torch.version.cuda = {torch.version.cuda}")
        torch.multiprocessing.set_start_method("spawn") # MAYBE NOT REQUIRED
    else:
        DEVICE = torch.device("cpu")
    # torch.set_default_device(DEVICE) # Requires newer PyTorch version
    print(f"DEVICE = {DEVICE}")

    return DEVICE