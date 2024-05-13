import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")

from src.setup import setup_python, setup_pytorch
from . import *

from src.dataset import MocaplabDatasetCNN
from .cnn.cnn import TestCNN
from src.dataset import MocaplabDatasetFC
from .fc.fc import MocaplabFC
from src.dataset import MocaplabDatasetLSTM
from .lstm.lstm import LSTM

def full_train() :

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()
    #DEVICE = torch.device("cpu")

    generator = torch.Generator()
    generator.manual_seed(0)

    # Dataset parameters
    
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None











    '''
    CNN Training
    '''
    # Training parameters
    BATCH_SIZE = 4                                  # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()     # Loss function
    OPTIMIZER_TYPE = "SGD"                          # Type of optimizer
    EPOCHS = [16, 999999]                           # Number of epochs
    LEARNING_RATES = [0.01, 0.001]                  # Learning rates
    EARLY_STOPPING = True                           # Early stopping flag
    PATIENCE = 999999                               # Early stopping patience
    MIN_DELTA = 0.0001                              # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### CNN Datasets ####")

    dataset = MocaplabDatasetCNN(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    # Split dataset
    n = len(dataset)
    diff = n - int(n*0.8) - 2*int(n*0.1)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.5), int(n*0.2), int(n*0.3)+diff], generator=generator)
    
    print(f"Total length -> {len(dataset)} samples")
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    
    # Data loaders
    print("#### CNN Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    # Create neural network
    print("#### CNN Model ####")
    model = TestCNN(nb_classes=2).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"CNN_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### CNN Training ####")

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
    test_acc, test_confusion_matrix, misclassified = test(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 misclassified)
    
    # Save model
    torch.save(model.state_dict(), "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/" + model_path + ".ckpt")
    
    # End training
    print("#### CNN End ####")













    '''
    Fully connected Training
    '''
    # Training parameters
    BATCH_SIZE = 4 # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer
    EPOCHS = [999999]                      # Number of epochs
    LEARNING_RATES = [0.001]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 999999        # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### FC Datasets ####")

    dataset = MocaplabDatasetFC(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    # Split dataset
    n = len(dataset)
    diff = n - int(n*0.8) - 2*int(n*0.1)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.5), int(n*0.2), int(n*0.3)+diff], generator=generator)
    
    print(f"Total length -> {len(dataset)} samples")
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    
    # Data loaders
    print("#### FC Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    # Create neural network
    print("#### FC Model ####")
    model = TestCNN(nb_classes=2).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"FC_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### FC Training ####")

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
    test_acc, test_confusion_matrix, misclassified = test(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 misclassified)
    
    # Save model
    torch.save(model.state_dict(), "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/" + model_path + ".ckpt")
    
    # End training
    print("#### FC End ####")













    '''
    LSTM Training
    '''
    # Training parameters
    BATCH_SIZE = 4 # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "Adam"                      # Type of optimizer
    EPOCHS = [16, 999999]                      # Number of epochs
    LEARNING_RATES = [0.0005, 0.0001]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 999999        # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### LSTM Datasets ####")

    dataset = MocaplabDatasetLSTM(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    # Split dataset
    n = len(dataset)
    diff = n - int(n*0.8) - 2*int(n*0.1)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.5), int(n*0.2), int(n*0.3)+diff], generator=generator)
    
    print(f"Total length -> {len(dataset)} samples")
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    
    # Data loaders
    print("#### LSTM Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    # Create neural network
    print("#### LSTM Model ####")
    model = LSTM(nb_classes=2).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"LSTM_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### LSTM Training ####")

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
    test_acc, test_confusion_matrix, misclassified = test(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 misclassified)
    
    # Save model
    torch.save(model.state_dict(), "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/" + model_path + ".ckpt")
    
    # End training
    print("#### LSTM End ####")