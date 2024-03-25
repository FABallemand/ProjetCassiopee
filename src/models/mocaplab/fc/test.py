import os
import sys
from datetime import datetime
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")
from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDataset
from mocaplab_fc import MocaplabFC
from plot_results import plot_results
from train import *

DEVICE = setup_pytorch()

def get_random_data(mocaplab_dataset) :
    i = random.randint(0, len(mocaplab_dataset.y))
    return mocaplab_dataset.__getitem__(i)

if __name__=='__main__':

    print("#### Datasets ####")

    dataset = MocaplabDataset(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    print("#### Model ####")
    model = MocaplabFC(dataset.max_length*237).to(DEVICE)

    print(os.getcwd())

    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab_fc/saved_models/model_20240322_114152.ckpt"))
    model = model.to(DEVICE)
    model = model.double()

    x, label = get_random_data(dataset)
    x = torch.tensor(x).to(DEVICE)
    label = torch.tensor(label).to(DEVICE)
    
    data_flattened = x.double().flatten()
    output = model(data_flattened.double())

    print(output, label)