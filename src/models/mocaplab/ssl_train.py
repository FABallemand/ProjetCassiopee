from datetime import datetime

import torch
from torch.utils.data import DataLoader

from train import *
from plot_results import *

from cnn.cnn_dataset import MocaplabDatasetCNN
from cnn.cnn import TestCNN, SSL_CNN
from fc.fc_dataset import MocaplabDatasetFC
from fc.fc import MocaplabFC, SSL_FC
from lstm.lstm_dataset import MocaplabDatasetLSTM
#from lstm.lstm import LSTM, SSL_LSTM

def plot_results(train_losses, validation_losses,
                 run_epochs, architecture, start_timestamp, device,
                 loss_function, optimizer_type, epochs,
                 learning_rates, early_stopping, patience, min_delta,
                 stop_timestamp, model_path):
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    # Compute total number of run epochs
    nb_epochs = sum(run_epochs)
    t = np.arange(nb_epochs)

    offset = 0
    for e in run_epochs:
        axs[0].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0].legend()

    # Plot loss over time
    axs[0].plot(t[1:], train_losses[1:], label="Train loss")
    axs[0].plot(t[1:], validation_losses[1:], label="Validation loss")
    offset = 0
    for e in run_epochs:
        axs[0].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0].legend()

    # Plot relevant data about the neural network and the training
    data = [
        ["Architecture", architecture],
        ["Start training", start_timestamp.strftime("%Y/%m/%d %H:%M:%S")],
        ["Device", device],
        ["Loss function", loss_function],
        ["Optimizer", optimizer_type],
        ["Epochs", epochs],
        ["Learning rate(s)", learning_rates],
        ["Run epochs", run_epochs if early_stopping else ""],
        ["Patience", patience if early_stopping else ""],
        ["Min delta", min_delta if early_stopping else ""],
        ["Stop training", stop_timestamp.strftime("%Y/%m/%d %H:%M:%S")]
    ]

    axs[1].table(cellText=data, loc="center")
    axs[1].axis("off")

    plt.show()

    # Save figure
    plt.savefig("self_supervised_learning/dev/ProjetCassiopee/train_results/mocaplab/" + model_path + ".png")



if __name__ == "__main__" :

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()
    #DEVICE = torch.device("cpu")
    

    # Dataset parameters
    
    NB_MAX_TRAIN_SAMPLES = None
    NB_MAX_VALIDATION_SAMPLES = None
    NB_MAX_TEST_SAMPLES = None




    '''
    Unsupervised CNN Training
    '''
    # Training parameters
    BATCH_SIZE = 2                                  # Batch size
    LOSS_FUNCTION = torch.nn.L1Loss()               # Loss function
    OPTIMIZER_TYPE = "SGD"                          # Type of optimizer
    EPOCHS = [32, 999999]                           # Number of epochs
    LEARNING_RATES = [0.05, 0.01]                   # Learning rates
    EARLY_STOPPING = True                           # Early stopping flag
    PATIENCE = 10                                   # Early stopping patience
    MIN_DELTA = 0.001                               # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # Datasets
    print("#### CNN Datasets ####")

    data_path = 'self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones'
    dataset = MocaplabDatasetCNN(data_path, padding=True)
    
    # Split dataset
    n = len(dataset)

    #SSL Datasets
    diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
    n_ssl = len(test_dataset)
    diff_ssl = n_ssl - int(n_ssl*0.8) - int(n_ssl*0.2)
    ssl_train_dataset, ssl_validation_dataset = torch.utils.data.random_split(test_dataset, lengths=[int(n_ssl*0.8), int(n_ssl*0.2) + diff_ssl], generator=generator)

    print(f"Total length -> {len(dataset)} samples")
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")

    print(f"SSL Train dataset -> {len(ssl_train_dataset)} samples")
    print(f"SSL Validation dataset -> {len(ssl_validation_dataset)} samples")

    
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
    
    ssl_train_data_loader = DataLoader(ssl_train_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    ssl_validation_data_loader = DataLoader(ssl_validation_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
    
    
    # Create neural network
    print("#### CNN Model ####")
    model = SSL_CNN(nb_classes=2).to(DEVICE)

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...

    model_path = f"SSL_CNN_90%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### CNN Training ####")

    encoder_layers_to_save = {'conv1_1': model.conv1_1.state_dict(),
                            'conv1_2': model.conv1_2.state_dict(),
                            'conv2_1': model.conv2_1.state_dict(),
                            'conv2_2': model.conv2_2.state_dict(),
                            'conv3_1': model.conv3_1.state_dict(),
                            'conv3_2': model.conv3_2.state_dict(),
                            'conv3_3': model.conv3_3.state_dict(),
                            'fc1': model.fc1.state_dict(),
                            'fc2': model.fc2.state_dict(),
                            'fc3': model.fc3.state_dict()
                            }
    
    decoder_layers_to_save = {'transconv1_1': model.transconv1_1.state_dict(),
                            'transconv1_2': model.transconv1_2.state_dict(),
                            'transconv1_3': model.transconv1_3.state_dict(),
                            'transconv2_1': model.transconv2_1.state_dict(),
                            'transconv2_2': model.transconv2_2.state_dict(),
                            'transconv3_1': model.transconv3_1.state_dict(),
                            'transconv3_2': model.transconv3_2.state_dict(),
                            'transfc1' : model.transfc1.state_dict(),
                            'transfc2' : model.transfc2.state_dict(),
                            'transfc3' : model.transfc3.state_dict()
                            }

    # Train model
    train_acc, train_loss, val_acc, val_loss, run_epochs = train_ssl(model,
                                                                 ssl_train_data_loader,
                                                                 ssl_validation_data_loader,
                                                                 LOSS_FUNCTION,
                                                                 OPTIMIZER_TYPE,
                                                                 EPOCHS,
                                                                 LEARNING_RATES,
                                                                 EARLY_STOPPING,
                                                                 PATIENCE,
                                                                 MIN_DELTA,
                                                                 DEVICE,
                                                                 DEBUG,
                                                                 model_type="CNN",
                                                                 model_path=model_path,
                                                                 encoder_layers_to_save=encoder_layers_to_save,
                                                                 decoder_layers_to_save=decoder_layers_to_save)
    
    # Save training time stop
    stop_timestamp = datetime.now()
    
    # Test model
    names = test_ssl_cnn(model, test_data_loader, DEVICE)
    print(names)

    # Plot results
    '''plot_results(train_acc, train_loss,
                 val_acc, val_loss,
                 run_epochs, type(model).__name__, start_timestamp, DEVICE,
                 LOSS_FUNCTION, OPTIMIZER_TYPE,
                 EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                 test_acc, test_confusion_matrix, stop_timestamp, model_path,
                 [])'''
    
    # Save model
    
    torch.save(encoder_layers_to_save, "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/encoder_" + model_path + ".ckpt")
    torch.save(decoder_layers_to_save, "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/decoder_" + model_path + ".ckpt")

    # End training

    print("#### Unsupervised CNN End ####")



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

    #10% data
    diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
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
    state_dict = torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/encoder_" + model_path + ".ckpt")
    
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
        param.requires_grad = True

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    model_path = f"SSL_CNN_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
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

    print("#### SSL CNN End ####")
