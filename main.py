import os
import numpy as np

import torch

import matplotlib.pyplot as plt

from src.setup import setup_python, setup_pytorch
from src.dataset import RGBDObjectDataset

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    setup_pytorch()

    # Datasets
    train_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                      train=True)
    
    test_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                     train=False)
    
    # Data loaders