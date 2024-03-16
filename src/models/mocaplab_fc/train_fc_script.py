import os
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ...setup import setup_python, setup_pytorch
from train import *
from ...dataset.mocaplab import MocaplabDataset
from fc import MocaplabFC


if __name__=='__main__':
    # Change working directory
    # abspath = os.path.abspath(__file__)
    # dname = os.path.dirname(abspath)
    # os.chdir(dname)

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
    OPTIMIZER_TYPE = "SGD"                     # Type of optimizer

    EPOCHS = [32, 16, 8, 4]                     # Number of epochs
    LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001] # Learning rates
    
    EARLY_STOPPING = False # Early stopping flag
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Datasets ####")

    dataset = MocaplabDataset(path="Cassiopée/Mocaplab_data",
                                    all_bones=False,
                                    train_test_ratio=8,
                                    validation_percentage=0.01)
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
    
    print(f"Train dataset -> {len(train_dataset.dataset.y)} samples")
    print(f"Test dataset -> {len(test_dataset.dataset.y)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    # Create neural network
    print("#### Model ####")

    model = FC(nb_classes=len(train_dataset.dataset.labels)).to(DEVICE)

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
                                                                 test_data_loader,
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