import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
import random

def create_optimizer(optimizer_type, model, learning_rate):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer type")

def train(
        model,
        train_data_loader,
        validation_data_loader,
        loss_function,
        optimizer_type,
        epochs_list,
        learning_rates_list,
        early_stopping=False,
        patience=5,
        min_delta=1e-3,
        device=torch.device("cpu"),
        debug=False,
        model_type="FC"):

    # Accuracies
    train_accuracies = []
    validation_accuracies = []

    # Losses
    train_losses = []
    validation_losses = []

    # Early stopping counter
    counter = 0
    bad_counter = 0 # Counter for stopping training if the model is not learning

    # Run epochs counter
    run_epochs = []

    for epochs, learning_rate in list(zip(epochs_list, learning_rates_list)):

        # Create optimizer
        optimizer = create_optimizer(optimizer_type, model, learning_rate)

        for epoch in range(epochs):
            print(f"#### EPOCH {epoch} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    if model_type == "CNN":
                        train_accuracy, train_loss = train_one_epoch_cnn(model, train_data_loader, loss_function, optimizer, device)
                    if model_type == "FC":
                        train_accuracy, train_loss = train_one_epoch_fc(model, train_data_loader, loss_function, optimizer, device)
                    if model_type == "LSTM":
                        train_accuracy, train_loss = train_one_epoch_lstm(model, train_data_loader, loss_function, optimizer, device)
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
            else:
                if model_type == "CNN":
                    train_accuracy, train_loss = train_one_epoch_cnn(model, train_data_loader, loss_function, optimizer, device)
                if model_type == "FC":
                    train_accuracy, train_loss = train_one_epoch_fc(model, train_data_loader, loss_function, optimizer, device)
                if model_type == "LSTM":
                    train_accuracy, train_loss = train_one_epoch_lstm(model, train_data_loader, loss_function, optimizer, device)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            # Evaluate model
            if model_type == "CNN":
                validation_accuracy, validation_loss = evaluate_cnn(model, validation_data_loader, loss_function, device)
            if model_type == "FC":
                validation_accuracy, validation_loss = evaluate_fc(model, validation_data_loader, loss_function, device)
            if model_type == "LSTM":
                validation_accuracy, validation_loss = evaluate_lstm(model, validation_data_loader, loss_function, device)
            validation_accuracies.append(validation_accuracy)
            validation_losses.append(validation_loss)

            print(f"Train:      accuracy={train_accuracy:.8f} | loss={train_loss:.8f}")
            print(f"Validation: accuracy={validation_accuracy:.8f} | loss={validation_loss:.8f}")

            # Early stopping
            if early_stopping:
                if epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta and validation_accuracy > 0.8:
                    counter += 1
                    if counter >= patience:
                        print("==== Early Stopping ====")
                        break
                elif epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta and validation_accuracy < 0.8 :
                    bad_counter += 1
                    if bad_counter >= 2*patience:
                        print("==== Not learning ====")
                        break
                        #exit(0)
                else:
                    counter = 0

        run_epochs.append(epoch + 1)

    return train_accuracies, train_losses, validation_accuracies, validation_losses, run_epochs


def train_one_epoch_fc(model, data_loader, loss_function, optimizer, device):

    # Enable training
    model.train(True)

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        data, label, name = batch
        data = data.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        data_flattened = data.view(data.size(0), -1)
        output = model(data_flattened)

        # Update accuracy variables
        _, predicted = torch.max(output.data, dim=1)

        total += len(label)
        batch_correct = (predicted == label).sum().item()
        correct += batch_correct

        # Compute loss
        loss = loss_function(output, label)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 5 == 0:
            # Batch loss
            print(f"    Batch {i:8}: accuracy={batch_correct / label.size(0):.4f} | loss={loss:.4f}")

    # Compute validation accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss

def evaluate_fc(model, data_loader, loss_function, device):

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            data_flattened = data.view(data.size(0), -1)
            output = model(data_flattened)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            # Compute loss
            loss = loss_function(output, label)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    if total != 0:
        validation_accuracy = correct / total
        validation_loss /= (i + 1) # Average loss over all batches of the epoch
    else :
        validation_accuracy = 0
        validation_loss = 0

    return validation_accuracy, validation_loss


def train_one_epoch_lstm(model, data_loader, loss_function, optimizer, device):
    
    # Enable training
    model.train(True)

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        data, label, name = batch
        data = data.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        output = model(data)

        # Update accuracy variables
        _, predicted = torch.max(output.data, dim=1)

        total += len(label)
        batch_correct = (predicted == label).sum().item()
        correct += batch_correct

        # Compute loss
        loss = loss_function(output, label)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 5 == 0:
            # Batch loss
            print(f"    Batch {i:8}: accuracy={batch_correct / label.size(0):.4f} | loss={loss:.4f}")

    # Compute validation accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss

def evaluate_lstm(model, data_loader, loss_function, device):
    
    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            # Compute loss
            loss = loss_function(output, label)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    if total != 0:
        validation_accuracy = correct / total
        validation_loss /= (i + 1) # Average loss over all batches of the epoch
    else :
        validation_accuracy = 0
        validation_loss = 0

    return validation_accuracy, validation_loss


def train_one_epoch_cnn(model, data_loader, loss_function, optimizer, device):
    
    # Enable training
    model.train(True)

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        data, label, name = batch
        data = data.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        output = model(data)

        # Update accuracy variables
        _, predicted = torch.max(output.data, dim=1)

        total += len(label)
        batch_correct = (predicted == label).sum().item()
        correct += batch_correct

        # Compute loss
        loss = loss_function(output, label)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 5 == 0:
            # Batch loss
            print(f"    Batch {i:8}: accuracy={batch_correct / label.size(0):.4f} | loss={loss:.4f}")

    # Compute validation accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss

def evaluate_cnn(model, data_loader, loss_function, device):
    
    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            # Compute loss
            loss = loss_function(output, label)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    if total != 0:
        validation_accuracy = correct / total
        validation_loss /= (i + 1)      # Average loss over all batches of the epoch
    else :
        validation_accuracy = 0
        validation_loss = 0

    return validation_accuracy, validation_loss


def test_fc(model, test_data_loader, device=torch.device("cpu")):
    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_label = None
    all_predicted = None

    misclassified = []

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            data_flattened = data.view(data.size(0), -1)
            output = model(data_flattened)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            for k in range(len(label)) :
                if label[k]!=predicted[k] :
                    misclassified.append((name[k], int(label[k].item())))

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = label.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, label))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_label.cpu(), all_predicted.cpu())

    return test_accuracy, test_confusion_matrix, misclassified


def test_lstm(model, test_data_loader, device=torch.device("cpu")):

    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_label = None
    all_predicted = None

    misclassified = []

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            for k in range(len(label)) :
                if label[k]!=predicted[k] :
                    misclassified.append((name[k], int(label[k].item())))

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = label.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, label))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_label.cpu(), all_predicted.cpu())

    return test_accuracy, test_confusion_matrix, misclassified


def test_cnn(model, test_data_loader, device=torch.device("cpu")):

    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_label = None
    all_predicted = None

    misclassified = []

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)

            # Update accuracy variables
            _, predicted = torch.max(output.data, dim=1)

            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

            for k in range(len(label)) :
                if label[k]!=predicted[k] :
                    misclassified.append((name[k], int(label[k].item())))

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = label.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, label))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_label.cpu(), all_predicted.cpu())

    return test_accuracy, test_confusion_matrix, misclassified




# --- Self-supervised learning ---

def train_ssl(
        model,
        train_data_loader,
        validation_data_loader,
        loss_function,
        optimizer_type,
        epochs_list,
        learning_rates_list,
        early_stopping=False,
        patience=5,
        min_delta=1e-3,
        device=torch.device("cpu"),
        debug=False,
        model_type="CNN",
        encoder_layers_to_save=None,
        decoder_layers_to_save=None,
        model_path=None) :
    
    # Accuracies
    train_accuracies = []
    validation_accuracies = []

    # Losses
    train_losses = []
    validation_losses = []

    # Early stopping counter
    counter = 0

    # Run epochs counter
    run_epochs = []

    for epochs, learning_rate in list(zip(epochs_list, learning_rates_list)):

        # Create optimizer
        optimizer = create_optimizer(optimizer_type, model, learning_rate)

        for epoch in range(epochs):
            print(f"#### EPOCH {epoch} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    if model_type == "CNN":
                        train_accuracy, train_loss = train_one_epoch_ssl_cnn(model, train_data_loader, loss_function, optimizer, device)
                    if model_type == "FC":
                        train_accuracy, train_loss = train_one_epoch_fc(model, train_data_loader, loss_function, optimizer, device)
                    if model_type == "LSTM":
                        train_accuracy, train_loss = train_one_epoch_lstm(model, train_data_loader, loss_function, optimizer, device)
                    train_losses.append(train_loss)
            else:
                if model_type == "CNN":
                        train_accuracy, train_loss = train_one_epoch_ssl_cnn(model, train_data_loader, loss_function, optimizer, device)
                if model_type == "FC":
                    train_accuracy, train_loss = train_one_epoch_ssl_fc(model, train_data_loader, loss_function, optimizer, device)
                if model_type == "LSTM":
                    train_accuracy, train_loss = train_one_epoch_ssl_lstm(model, train_data_loader, loss_function, optimizer, device)
                train_losses.append(train_loss)

            # Evaluate model
            if model_type == "CNN":
                validation_accuracy, validation_loss = evaluate_ssl_cnn(model, validation_data_loader, loss_function, device)
            if model_type == "FC":
                validation_accuracy, validation_loss = evaluate_ssl_fc(model, validation_data_loader, loss_function, device)
            if model_type == "LSTM":
                validation_accuracy, validation_loss = evaluate_ssl_lstm(model, validation_data_loader, loss_function, device)
            validation_losses.append(validation_loss)

            print(f"Train:      loss={train_loss:.8f}")
            print(f"Validation: loss={validation_loss:.8f}")

            # Plot visualisation of reconstruction
            if epoch % 5 == 0 and epoch <= 5 and learning_rate == learning_rates_list[0] :
                # Select random batch
                i = random.randint(0, len(validation_data_loader)-1)
                for j, batch in enumerate(validation_data_loader):
                    if j == i:
                        break

                data, label, name = batch
                data = data.to(torch.float32).to(device)
                if model_type == "FC":
                    data = data.view(data.size(0), -1)
                    output = model(data)
                    data = data.view(data.size(0), 1, 100, 237)
                    output = output.view(output.size(0), 1, 100, 237)
                else :
                    output = model(data)
                fig, axs = plt.subplots(2, 1, figsize=(16, 9))
                k = random.randint(0, len(data)-1)
                if type(axs) == np.ndarray:
                    axs[0].imshow(data[k][0].cpu().detach().numpy())
                    axs[1].imshow(output[k][0].cpu().detach().numpy())
                    axs[0].set_title(f"Input, {name[k]}")
                    axs[1].set_title("Output, " + name[k])
                '''else :
                    axs[k, 0].imshow(data[k][0].cpu().detach().numpy())
                    axs[k, 1].imshow(output[k][0].cpu().detach().numpy())
                    axs[k, 0].set_title(f"Input, {name[k]}")
                    axs[k, 1].set_title("Output, " + name[k])'''

                plt.show()


            # Early stopping
            if early_stopping:
                if epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta:
                    counter += 1
                    if counter >= patience:
                        print("==== Early Stopping ====")

                        # Plot visualisation of reconstruction bofore ending
                        # Select random batch
                        i = random.randint(0, len(validation_data_loader)-1)
                        for j, batch in enumerate(validation_data_loader):
                            if j == i:
                                break

                        data, label, name = batch
                        data = data.to(torch.float32).to(device)
                        if model_type == "FC":
                            data = data.view(data.size(0), -1)
                            output = model(data)
                            data = data.view(data.size(0), 1, 100, 237)
                            output = output.view(output.size(0), 1, 100, 237)
                        else :
                            output = model(data)
                        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
                        k = random.randint(0, len(data)-1)
                        if type(axs) == np.ndarray:
                            axs[0].imshow(data[k][0].cpu().detach().numpy())
                            axs[1].imshow(output[k][0].cpu().detach().numpy())
                            axs[0].set_title(f"Input, {name[k]}")
                            axs[1].set_title("Output, " + name[k])
                        
                        break
                else:
                    counter = 0
            
            if epoch % 25 == 0 and epoch != 0 :
                torch.save(encoder_layers_to_save, "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/encoder_" + model_path + f"_epoch_{epoch}.ckpt")
                torch.save(decoder_layers_to_save, "self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/decoder_" + model_path + f"_epoch_{epoch}.ckpt")
                print(f"Model saved at epoch {epoch}")

        run_epochs.append(epoch + 1)
    
    return train_accuracies, train_losses, validation_accuracies, validation_losses, run_epochs


def train_one_epoch_ssl_cnn(
        model,
        data_loader,
        loss_function,
        optimizer,
        device) :
    
    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        data, label, name = batch
        data = data.to(torch.float32).to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        output = model(data)

        # Compute loss
        loss = loss_function(output, data)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 5 == 0:
            # Batch loss
            print(f"    Batch {i:8}: loss={loss:.4f}")

    # Compute validation accuracy and loss
    train_accuracy = 0
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss

def evaluate_ssl_cnn(
        model,
        data_loader,
        loss_function,
        device) :
    
    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)

            # Compute loss
            loss = loss_function(output, data)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    validation_accuracy = 0
    validation_loss /= (i + 1)
    return validation_accuracy, validation_loss

def test_ssl_cnn(model, test_data_loader, device=torch.device("cpu")) :
    """
    Unused function
    """

    i1, i2, i3 = random.randint(0, len(test_data_loader.dataset)-1), random.randint(0, len(test_data_loader.dataset)-1), random.randint(0, len(test_data_loader.dataset)-1)

    with torch.no_grad():

        names = []

        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)

            # Make predictions for batch
            output = model(data)
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            axs[0].imshow(data[0][0].cpu().numpy())
            axs[1].imshow(output[0][0].cpu().numpy())
            plt.show()
            
    return names


def train_one_epoch_ssl_fc(
        model,
        data_loader,
        loss_function,
        optimizer,
        device) :
    
    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        data, label, name = batch
        data = data.to(torch.float32).to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        data_flattened = data.view(data.size(0), -1)
        output = model(data_flattened)

        # Compute loss
        loss = loss_function(output, data_flattened)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 5 == 0:
            # Batch loss
            print(f"    Batch {i:8}: loss={loss:.4f}")

    # Compute validation accuracy and loss
    train_accuracy = 0
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss

def evaluate_ssl_fc(
        model,
        data_loader,
        loss_function,
        device) :
    
    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            
            # Load and prepare batch
            data, label, name = batch
            data = data.to(torch.float32).to(device)

            # Make predictions for batch
            data_flattened = data.view(data.size(0), -1)
            output = model(data_flattened)

            # Compute loss
            loss = loss_function(output, data_flattened)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    validation_accuracy = 0
    validation_loss /= (i + 1)
    return validation_accuracy, validation_loss


def train_one_epoch_ssl_lstm() :
    pass

def evaluate_ssl_lstm() :
    pass
