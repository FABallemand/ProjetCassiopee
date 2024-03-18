import os
import sys
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")
from src.setup import setup_python, setup_pytorch
from src import plot_results

from src.train import create_optimizer
from src.models.autoencoder.autoencoder import TestAutoencoder
from src.models.combined_model.train_contrastive import contrastive_loss
from src.models.unsupervised.train_unsupervised import train

from src.dataset.rgbd_objects_unsupervised import RGBDObjectDataset, RGBDObjectDataset_Contrast


if __name__=='__main__':

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (256,256)
    TRANSFORMATION = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size=INPUT_SIZE)])
    NB_TRAIN_SAMPLES = None
    NB_VALIDATION_SAMPLES = None
    NB_TEST_SAMPLES = None

    # Training parameters
    BATCH_SIZE = 1 # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer
    EPOCHS = [1000]         # Number of epochs
    LEARNING_RATES = [0.01] # Learning rates
    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Datasets ####")

    train_dataset = RGBDObjectDataset_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                               mode="train",
                                               transformation=TRANSFORMATION,
                                               nb_samples=NB_TRAIN_SAMPLES)
    
    validation_dataset = RGBDObjectDataset_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                                    mode="validation",
                                                    transformation=TRANSFORMATION,
                                                    nb_samples=NB_VALIDATION_SAMPLES)
    
    test_dataset = RGBDObjectDataset_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                              mode="test",
                                              transformation=TRANSFORMATION,
                                              nb_samples=NB_TEST_SAMPLES)
    
    print(f"Train dataset -> {len(train_dataset.y)} samples")
    print(f"Validation dataset -> {len(validation_dataset.y)} samples")
    print(f"Test dataset -> {len(test_dataset.y)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    # Create neural network
    print("#### Model ####")

    model = TestAutoencoder().to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"train_results/model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### Training ####")

    # Train model
    train_loss = train(model, train_data_loader, OPTIMIZER_TYPE, EPOCHS, LEARNING_RATES, DEVICE, DEBUG)
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # End training
    print("#### End ####")