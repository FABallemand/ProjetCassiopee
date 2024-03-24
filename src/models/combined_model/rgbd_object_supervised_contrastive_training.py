import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchview

from ...setup import setup_python, setup_pytorch
from ...plot import plot_summary
from ...transformation import RandomCrop, ObjectCrop
from ...dataset import RGBDObjectDataset_Supervised_Contrast
from .combined_model import CombinedModel
from .train_contrastive import train, test

def rgbd_object_combined_supervised_contrastive_training():

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (256,256)
    MODALITIES = ["rgb"]
    TRANSFORMATION = None
    CROP_TRANSFORMATION = ObjectCrop(output_size=INPUT_SIZE,
                                     padding=(20,20),
                                     offset_range=(-10,10))
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None

    # Training parameters
    BATCH_SIZE = 64   # Batch size
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

    print("## Train Dataset ##")
    train_dataset = RGBDObjectDataset_Supervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                                          mode="train",
                                                          modalities=MODALITIES,
                                                          transformation=TRANSFORMATION,
                                                          crop_transformation=CROP_TRANSFORMATION,
                                                          nb_max_samples=NB_MAX_TRAIN_SAMPLES)
    
    print("## Validation Dataset ##")
    validation_dataset = RGBDObjectDataset_Supervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                                               mode="validation",
                                                               modalities=MODALITIES,
                                                               transformation=TRANSFORMATION,
                                                               crop_transformation=CROP_TRANSFORMATION,
                                                               nb_max_samples=NB_MAX_VALIDATION_SAMPLES)
    
    print("## Test Dataset ##")
    test_dataset = RGBDObjectDataset_Supervised_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                                         mode="test",
                                                         modalities=MODALITIES,
                                                         transformation=TRANSFORMATION,
                                                         crop_transformation=CROP_TRANSFORMATION,
                                                         nb_max_samples=NB_MAX_TEST_SAMPLES)
    
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")

    print("## Train Data Loader ##")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=SHUFFLE,
                                   drop_last=DROP_LAST)
    
    print("## Validation Data Loader ##")
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=SHUFFLE,
                                        drop_last=DROP_LAST)
    
    print("## Test Data Loader ##")
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=SHUFFLE,
                                  drop_last=DROP_LAST)
    
    # Create neural network
    print("#### Model ####")

    model = CombinedModel().to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    results_dir = f"train_results/supervised_contrastive"
    results_file = f"rgbd_object_combined_model_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

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
    test_acc, test_confusion_matrix, tsne_results_2d, tsne_results_3d, labels = test(model, test_data_loader, True, DEVICE)

    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, results_file))

    # Plot results
    plot_summary(type(train_dataset).__name__, INPUT_SIZE, None, MODALITIES,
                 type(TRANSFORMATION).__name__, type(CROP_TRANSFORMATION).__name__,
                 len(train_dataset), len(validation_dataset), len(test_dataset),
                 BATCH_SIZE, SHUFFLE, DROP_LAST,
                 DEVICE, type(model).__name__, DEBUG,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES,
                 EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 start_timestamp, stop_timestamp, run_epochs,
                 train_acc, train_loss,
                 val_acc, val_loss,
                 test_acc, test_confusion_matrix,
                 tsne_results_2d, tsne_results_3d, labels,
                 os.path.join(results_dir, results_file + "_res.png"))
    
    # Plot model architecture
    graph = torchview.draw_graph(model, input_size=(BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]), device=DEVICE,
                                 save_graph=True, filename=results_file + "_arc", directory=results_dir)
    
    # End training
    print("#### End ####")