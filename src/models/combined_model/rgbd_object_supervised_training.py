import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchview

from ...setup import setup_python, setup_pytorch
from ...train import plot_results
from ...transformation import RandomCrop, ObjectCrop
from ...dataset import RGBDObjectDataset
from .combined_model import CombinedModel
from .train import train, test


def rgbd_object_combined_supervised_training():

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (256,256)
    TRANSFORMATION = None
    CROP_TRANSFORMATION = ObjectCrop(output_size=INPUT_SIZE,
                                     padding=(20,20),
                                     offset_range=(-10,10))
    NB_TRAIN_SAMPLES = None
    NB_VALIDATION_SAMPLES = None
    NB_TEST_SAMPLES = None

    # Training parameters
    BATCH_SIZE = 64 # Batch size

    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer

    EPOCHS = [1]         # Number of epochs
    LEARNING_RATES = [0.01] # Learning rates
    
    EARLY_STOPPING = False # Early stopping flag
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum delta

    DEBUG = False # Debug flag
    
    # Datasets
    print("#### Datasets ####")
    
    print("## Train Dataset ##")
    train_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                      mode="train",
                                      modalities=["rgb"],
                                      transformation=TRANSFORMATION,
                                      crop_transformation=CROP_TRANSFORMATION,
                                      nb_samples=NB_TRAIN_SAMPLES)
    
    print("## Validation Dataset ##")
    validation_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                           mode="validation",
                                           modalities=["rgb"],
                                           transformation=TRANSFORMATION,
                                           crop_transformation=CROP_TRANSFORMATION,
                                           nb_samples=NB_VALIDATION_SAMPLES)
    
    print("## Test Dataset ##")
    test_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                     mode="test",
                                     modalities=["rgb"],
                                     transformation=TRANSFORMATION,
                                     crop_transformation=CROP_TRANSFORMATION,
                                     nb_samples=NB_TEST_SAMPLES)
    
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")

    print("## Train Data Loader ##")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    
    print("## Validation Data Loader ##")
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    
    print("## Test Data Loader ##")
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    # Create neural network
    print("#### Model ####")

    model = CombinedModel().to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    results_dir = f"train_results/supervised"
    results_file = f"rgbd_object_cnn_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

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
    test_acc, test_confusion_matrix = test(model, test_data_loader, os.path.join(results_dir, results_file + "_tsne.png"), DEVICE)

    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, results_file))

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, os.path.join(results_dir, results_file + "_res"))
    
    # Plot model architecture
    graph = torchview.draw_graph(model, input_size=(BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), device=DEVICE,
                                 save_graph=True, filename=results_file + "_arc", directory=results_dir)
    
    # End training
    print("#### End ####")