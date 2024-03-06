import os
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from src.setup import setup_python, setup_pytorch
from src.dataset import RGBDObjectDataset
from src.models import TestCNN
from src.train import train, test, plot_results

# Run with: nohup python3 main.py &

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Begin set-up
    print("#### Start Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()
    # DEVICE = torch.device("cpu")

    # Dataset parameters
    TRANSFORM = None
    # TRANSFORM = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    NB_TRAIN_IMGS = 100
    NB_TEST_IMGS = 10

    # Training parameters
    BATCH_SIZE = 8 # Batch size

    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "Adam"                     # Type of optimizer

    EPOCHS = [10, 10, 5]                     # Number of epochs
    LEARNING_RATES = [0.001, 0.0001, 0.00001] # Learning rates
    
    EARLY_STOPPING = False # Early stopping flag
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Start Datasets Creation ####")

    train_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                      mode="train",
                                      transform=TRANSFORM,
                                      nb_imgs=NB_TRAIN_IMGS)
    
    test_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                     mode="test",
                                     transform=TRANSFORM,
                                     nb_imgs=NB_TEST_IMGS)
    
    # Data loaders
    print("#### Start Data Loaders Creation ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    # Create neural network
    print("#### Start Creating Model ####")

    model = TestCNN(nb_classes=len(train_dataset.class_dict)).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"model/model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    # model_path = f"test/model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### Start Training ####")

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
    test_acc, test_confusion_matrix = test(model, test_data_loader, DEVICE, DEBUG)

    # Plot results
    plot_results(train_acc, train_loss, val_acc, val_loss, run_epochs,
                 type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # End training
    print("#### End Training ####")