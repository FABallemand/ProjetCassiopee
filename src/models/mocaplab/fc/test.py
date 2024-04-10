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
from src.dataset import MocaplabDatasetFC
from mocaplab import MocaplabFC
from plot_results import plot_results
from train import *

DEVICE = setup_pytorch()

def get_random_data(mocaplab_dataset) :
    i = random.randint(0, len(mocaplab_dataset.y))
    return mocaplab_dataset.__getitem__(i)

if __name__=='__main__':

    print("#### Datasets ####")

    dataset = MocaplabDatasetFC(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)
    
    print("#### Model ####")
    model = MocaplabFC(dataset.max_length*237).to(DEVICE)

    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/fc/saved_models/model_20240325_141951.ckpt"))
    model = model.to(DEVICE)
    model = model.double()

    for batch in data_loader :

        data, label = batch
        data = data.to(DEVICE)
        label = label.to(DEVICE)
    
        data_flattened = data.view(data.size(0), -1)
        output = model(data_flattened.double())

        _, predicted = torch.max(output.data, dim=1)

        print("Predicted : ", predicted.data, "/ Label : ", label.data)