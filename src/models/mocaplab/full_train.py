from datetime import datetime

import torch
from torch.utils.data import DataLoader

from train import *
from plot_results import *

from cnn.cnn_dataset import MocaplabDatasetCNN
from cnn.cnn import TestCNN
from fc.fc_dataset import MocaplabDatasetFC
from fc.fc import MocaplabFC
from lstm.lstm_dataset import MocaplabDatasetLSTM
from lstm.lstm import LSTM

if __name__ == "__main__" :

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None









    
    '''
    Fully connected Training
    '''
    # Training parameters
    BATCH_SIZE = 2 # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer
    EPOCHS = [999999]                      # Number of epochs
    LEARNING_RATES = [0.1]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 10        # Early stopping patience
    MIN_DELTA = 0.001     # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # Datasets
    print("#### FC Datasets ####")

    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetFC(data_path, padding=True)
    
    # Split dataset
    n = len(dataset)

    diff = n - int(n*0.6) - int(n*0.1) - int(n*0.3)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.6), int(n*0.1), int(n*0.3)+diff], generator=generator)

    #50% data
    #diff = n - int(n*0.3) - int(n*0.2) - int(n*0.5)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #25% data
    #diff = n - int(n*0.15) - int(n*0.1) - int(n*0.75)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #10% data
    #diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
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
    model = MocaplabFC().to(DEVICE)

    """state_dict = torch.load("self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones")
    
    flattened_state_dict = {}
    for key, val in state_dict.items():
        for sub_key, sub_val in val.items():
            new_key = key + '.' + sub_key
            flattened_state_dict[new_key] = sub_val
    
    model.load_state_dict(state_dict=flattened_state_dict)

    # Désactiver le calcul du gradient pour tous les paramètres du modèle
    for param in model.parameters():
        param.requires_grad = False

    # Activer le calcul du gradient pour les paramètres de fc1, fc2, fc3
    for param in model.fc1.parameters():
        param.requires_grad = True
    #for param in model.fc2.parameters():
    #    param.requires_grad = True
    #for param in model.fc3.parameters():
    #    param.requires_grad = True"""

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving the model and results
    model_path = f"FC_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"SSL_FC_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

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
                                                                 DEBUG,
                                                                 model_type="FC")
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Test model
    test_acc, test_confusion_matrix, misclassified = test_fc(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 [])
                 #misclassified)
    
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
    EPOCHS = [999999]                      # Number of epochs
    LEARNING_RATES = [0.0004]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 10        # Early stopping patience
    MIN_DELTA = 0.01     # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # Datasets
    print("#### LSTM Datasets ####")

    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetLSTM(path=data_path,
                                    return_filename=True,
                                    padding = True)
    
    # Split dataset
    n = len(dataset)

    diff = n - int(n*0.6) - int(n*0.1) - int(n*0.3)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.6), int(n*0.1), int(n*0.3)+diff], generator=generator)

    #50% data
    #diff = n - int(n*0.3) - int(n*0.2) - int(n*0.5)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #25% data
    #diff = n - int(n*0.15) - int(n*0.1) - int(n*0.75)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #10% data
    diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
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
    model = LSTM(input_size=237, hidden_size=48, num_layers=4, output_size=2).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"LSTM_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"LSTM_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"LSTM_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"LSTM_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

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
                                                                 DEBUG,
                                                                 model_type="LSTM")
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Test model
    test_acc, test_confusion_matrix, misclassified = test_lstm(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 [])
    
    # Save model
    torch.save(model.state_dict(), "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/" + model_path + ".ckpt")
    
    # End training
    print("#### LSTM End ####")

    










    '''
    CNN Training
    '''
    # Training parameters
    BATCH_SIZE = 2                                  # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()     # Loss function
    OPTIMIZER_TYPE = "SGD"                          # Type of optimizer
    EPOCHS = [16, 999999]                           # Number of epochs
    LEARNING_RATES = [0.02, 0.001]                  # Learning rates
    EARLY_STOPPING = True                           # Early stopping flag
    PATIENCE = 10                               # Early stopping patience
    MIN_DELTA = 0.01                              # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # Datasets
    print("#### CNN Datasets ####")

    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetCNN(data_path, padding=True)
    
    # Split dataset
    n = len(dataset)

    diff = n - int(n*0.6) - int(n*0.1) - int(n*0.3)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.6), int(n*0.1), int(n*0.3)+diff], generator=generator)

    #50% data
    #diff = n - int(n*0.3) - int(n*0.2) - int(n*0.5)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #25% data
    #diff = n - int(n*0.15) - int(n*0.1) - int(n*0.75)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #10% data
    #diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
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
    """state_dict = torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/SSL_CNN/encoder_SSL_CNN_90%_20240529_162101_epoch_280.ckpt")
    
    flattened_state_dict = {}
    for key, val in state_dict.items():
        for sub_key, sub_val in val.items():
            new_key = key + '.' + sub_key
            flattened_state_dict[new_key] = sub_val
    
    model.load_state_dict(state_dict=flattened_state_dict)

    # Désactiver le calcul du gradient pour tous les paramètres du modèle
    for param in model.parameters():
        param.requires_grad = False

    # Activer le calcul du gradient pour les paramètres de fc1 et fc2
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True
    for param in model.fc3.parameters():
        param.requires_grad = True
    for param in model.conv3_3.parameters():
        param.requires_grad = True
    for param in model.conv3_2.parameters():
        param.requires_grad = True
    for param in model.conv3_1.parameters():
        param.requires_grad = True"""

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"CNN_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"CNN_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"CNN_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"CNN_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"SSL_CNN_fc-only_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

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
                                                                 DEBUG,
                                                                 model_type="CNN")
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Test model
    test_acc, test_confusion_matrix, misclassified = test_cnn(model, test_data_loader, DEVICE)

    # Plot results
    plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 [])
    
    # Save model
    torch.save(model.state_dict(), "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/" + model_path + ".ckpt")
    
    # End training

    print("#### CNN End ####")
    
