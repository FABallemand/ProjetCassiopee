import os
from datetime import datetime
import logging

import torch
from torch.utils.data import DataLoader

from ....setup import setup_python, setup_pytorch
from ....transformation import ObjectCrop
from ....dataset import RGBDObjectDataset
from .autoencoder import ResNetAutoencoder
from .train import train, test


def rgbd_object_ae_unsupervised_training():

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    results_dir = f"train_results/rgbd_object/unsupervised/ae_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    os.mkdir(results_dir)

    # Configure logging
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(filename=os.path.join(results_dir, "training.log"),
                        level=logging.DEBUG, format=FORMAT)

    # Begin set-up
    logging.info("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (256,256) # Width and hheight of the imputs
    MODALITIES = ["rgb"]   # Modalities
    TRANSFORMATION = None  # Transformation of the inputs
    CROP_TRANSFORMATION = ObjectCrop(output_size=INPUT_SIZE,
                                     padding=(20,20),
                                     offset_range=(-10,10))
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None
    # NB_MAX_TRAIN_SAMPLES = 20000
    # NB_MAX_VALIDATION_SAMPLES = 100
    # NB_MAX_TEST_SAMPLES = None
    SPLIT = 1 # Split of the dataset (None, 0, 1)

    # Training parameters
    WEIGHTS_FREEZING = False # Weight freezing
    LAST_CHECKPOINT = None   # Last checkpoint to load

    BATCH_SIZE = 50   # Batch size
    SHUFFLE = True    # Shuffle
    DROP_LAST = False # Drop last batch
    NUM_WORKERS = 0   # Number of prpocesses
    PIN_MEMORY = True # Memory pinning

    LOSS_FUNCTION = torch.nn.MSELoss() # Loss function
    OPTIMIZER_TYPE = "SGD"             # Type of optimizer

    EPOCHS = [1000]          # Number of epochs
    LEARNING_RATES = [0.001] # Learning rates
    
    EARLY_STOPPING = False # Early stopping
    PATIENCE = 10          # Early stopping patience
    MIN_DELTA = 0.0001     # Early stopping minimum deltaCwewKWRvMjQQdb5l

    DEBUG = False # Debug flag
    
    # Training datasets
    logging.info("#### Datasets ####")

    logging.info(f"INPUT_SIZE = {INPUT_SIZE}")
    logging.info(f"MODALITIES = {MODALITIES}")
    logging.info(f"TRANSFORMATION = {TRANSFORMATION}")
    logging.info(f"CROP_TRANSFORMATION = {CROP_TRANSFORMATION}")
    logging.info(f"NB_MAX_TRAIN_SAMPLES = {NB_MAX_TRAIN_SAMPLES}")
    logging.info(f"NB_MAX_VALIDATION_SAMPLES = {NB_MAX_VALIDATION_SAMPLES}")
    logging.info(f"NB_MAX_TEST_SAMPLES = {NB_MAX_TEST_SAMPLES}")
    logging.info(f"SPLIT = {SPLIT}")
    
    logging.info("## Train Dataset ##")
    train_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                      mode="train",
                                      modalities=MODALITIES,
                                      transformation=TRANSFORMATION,
                                      crop_transformation=CROP_TRANSFORMATION,
                                      nb_max_samples=NB_MAX_TRAIN_SAMPLES,
                                      split=SPLIT)
    logging.info(f"{len(train_dataset)} samples")

    logging.info("## Validation Dataset ##")
    validation_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                           mode="validation",
                                           modalities=MODALITIES,
                                           transformation=TRANSFORMATION,
                                           crop_transformation=CROP_TRANSFORMATION,
                                           nb_max_samples=NB_MAX_VALIDATION_SAMPLES,
                                           split=SPLIT)
    logging.info(f"{len(validation_dataset)} samples")
    
    # Training data loaders
    logging.info("#### Data Loaders ####")

    logging.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logging.info(f"SHUFFLE = {SHUFFLE}")
    logging.info(f"DROP_LAST = {DROP_LAST}")
    logging.info(f"NUM_WORKERS = {NUM_WORKERS}")
    logging.info(f"PIN_MEMORY = {PIN_MEMORY}")

    logging.info("## Train Data Loader ##")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=SHUFFLE,
                                   drop_last=DROP_LAST,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY)
    
    
    logging.info("## Validation Data Loader ##")
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=SHUFFLE,
                                        drop_last=DROP_LAST,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    
    # Neural network
    logging.info("#### Model ####")

    logging.info(f"WEIGHTS_FREEZING = {WEIGHTS_FREEZING}")
    logging.info(f"LAST_CHECKPOINT = {LAST_CHECKPOINT}")
    logging.info(f"LOSS_FUNCTION = {LOSS_FUNCTION}")
    logging.info(f"OPTIMIZER_TYPE = {OPTIMIZER_TYPE}")
    logging.info(f"WEIGHTS_FREEZING = {WEIGHTS_FREEZING}")
    logging.info(f"EPOCHS = {EPOCHS}")
    logging.info(f"LEARNING_RATES = {LEARNING_RATES}")
    logging.info(f"EARLY_STOPPING = {EARLY_STOPPING}")
    logging.info(f"PATIENCE = {PATIENCE}")
    logging.info(f"MIN_DELTA = {MIN_DELTA}")
    logging.info(f"DEBUG = {DEBUG}")

    # Create model
    model = ResNetAutoencoder(WEIGHTS_FREEZING)

    # Load last checkpoint if specified
    if LAST_CHECKPOINT is not None and os.path.isfile(LAST_CHECKPOINT):
        model.load_state_dict(torch.load(LAST_CHECKPOINT))

    # Load model to PyTorch device
    model = model.to(DEVICE)

    # Print model
    logging.info(model)

    # Training
    logging.info("#### Training ####")

    # Train model
    train_loss, val_loss, run_epochs = train(model,
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
                                             results_dir,
                                             DEBUG)
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Testing
    logging.info("#### Testing ####")

    # Testing dataset
    logging.info("## Test Dataset ##")
    test_dataset = RGBDObjectDataset(path="data/RGB-D_Object/rgbd-dataset",
                                     mode="test",
                                     modalities=MODALITIES,
                                     transformation=TRANSFORMATION,
                                     crop_transformation=CROP_TRANSFORMATION,
                                     nb_max_samples=NB_MAX_TEST_SAMPLES,
                                     split=SPLIT)
    logging.info(f"{len(test_dataset)} samples")

    # Testing data loader
    logging.info("## Test Data Loader ##")
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=SHUFFLE,
                                  drop_last=DROP_LAST,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=PIN_MEMORY)

    # Test model
    tsne_results_2d, tsne_results_3d, labels_arr = test(model, test_data_loader, True, DEVICE)
    
    # End training
    logging.info("#### End ####")