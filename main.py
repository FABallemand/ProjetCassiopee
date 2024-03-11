import os
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from src.setup import setup_python, setup_pytorch
from src.train import plot_results
from src.dataset.rgbd_objects import RGBDObjectDataset
from src.models.cnn import TestCNN, train, test
from src.models.autoencoder import TestAutoencoder, train, test

# Run with: nohup python3 main.py &

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (128,128)
    TRANSFORMATION = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size=INPUT_SIZE)])
    NB_TRAIN_SAMPLES = None
    NB_VALIDATION_SAMPLES = None
    NB_TEST_SAMPLES = None

    # Training parameters
    BATCH_SIZE = 128 # Batch size

    # LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    LOSS_FUNCTION = torch.nn.MSELoss() # Loss function
    OPTIMIZER_TYPE = "Adam"                     # Type of optimizer

    EPOCHS = [32, 16, 8, 4]                     # Number of epochs
    LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001] # Learning rates
    
    EARLY_STOPPING = False # Early stopping flag
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Datasets ####")

    train_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                      mode="train",
                                      transformation=TRANSFORMATION,
                                      nb_samples=NB_TRAIN_SAMPLES)
    
    validation_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                           mode="validation",
                                           transformation=TRANSFORMATION,
                                           nb_samples=NB_VALIDATION_SAMPLES)
    
    test_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
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

    # model = TestCNN(nb_classes=len(train_dataset.class_dict)).to(DEVICE)

    model = TestAutoencoder().to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"models/model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    # model_path = f"test/model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### Training ####")

    # Train model
    train_acc, train_loss, val_acc, val_loss, run_epochs = train(model,
                                                                 train_data_loader,
                                                                 validation_data_loader,
                                                                 LOSS_FUNCTION,
                                                                 OPTIMIZER_TYPE,
                                                                 EPOCHS,
                                                                 LEARNING_RATES,
                                                                 EARLY_STOPPING,
                                                                 PATIENCE,
                                                                 MIN_DELTA,
                                                                 DEVICE,
                                                                 DEBUG)
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Test model
    test_acc, test_confusion_matrix = test(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # End training
    print("#### End ####")