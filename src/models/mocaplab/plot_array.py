import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/")
from src.dataset import MocaplabDatasetFC
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatasetFC
from src.models.mocaplab import MocaplabFC
from fc.plot_results import plot_results
from fc.train import *

# Create figure
#fig, axs = plt.subplots(1, 1, figsize=(16, 9))
dataset = MocaplabDatasetFC(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
data_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)

# Get a sample
for i, (x, y) in enumerate(data_loader):
    x = x.squeeze().numpy()
    fig = plt.figure()
    plt.matshow(x)
    plt.savefig("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/array/array.png")
    plt.show()
    break

print('done')
