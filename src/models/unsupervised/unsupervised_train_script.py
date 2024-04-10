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
from src.plot import final_plot
from src.transformation import RandomCrop, ObjectCrop
from src.train import create_optimizer
from src.models.rgbd_object.autoencoder.autoencoder import TestAutoencoder
from src.loss.contrastive_loss import contrastive_loss
from src.models.unsupervised.test_simple_train_unsupervised import train
#from src.models.unsupervised.train_unsupervised import train
#from src.models.rgbd_object.autoencoder.rgbd_object_unsupervised_training import train
#from src.models.rgbd_object.autoencoder.rgbd_object_unsupervised_contrastive_training import train

from src.dataset.rgbd_objects import RGBDObjectDataset, RGBDObjectDataset_Unsupervised_Contrast


if __name__=='__main__':

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (256,256)
    MODALITIES = ["rgb"]
    TRANSFORMATION = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size=INPUT_SIZE)])
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None
    CROP_TRANSFORMATION = ObjectCrop(output_size=INPUT_SIZE,
                                     padding=(20,20),
                                     offset_range=(-10,10))
    
    # Training parameters
    BATCH_SIZE = 10 # Batch size
    SHUFFLE = True    # Shuffle
    DROP_LAST = False # Drop last batch
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer
    EPOCHS = [1]         # Number of epochs
    LEARNING_RATES = [0.01] # Learning rates
    EARLY_STOPPING = False # Early stopping
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Datasets ####")

    train_dataset = RGBDObjectDataset_Unsupervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                               mode="train",
                                               transformation=TRANSFORMATION,
                                               nb_max_samples=NB_MAX_TRAIN_SAMPLES)
    
    validation_dataset = RGBDObjectDataset_Unsupervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                                    mode="validation",
                                                    transformation=TRANSFORMATION,
                                                    nb_max_samples=NB_MAX_VALIDATION_SAMPLES)
    
    test_dataset = RGBDObjectDataset_Unsupervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                              mode="test",
                                              transformation=TRANSFORMATION,
                                              nb_max_samples=NB_MAX_TEST_SAMPLES)
    
    print(f"Train dataset -> {len(train_dataset.y)} samples")
    print(f"Validation dataset -> {len(validation_dataset.y)} samples")
    print(f"Test dataset -> {len(test_dataset.y)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=True)
    
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