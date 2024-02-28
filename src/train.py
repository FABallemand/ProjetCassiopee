import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def create_optimizer(optimizer_type, model, learning_rate):
    """
    Create optimizer based on the PyTorch name of the optimizer.

    Parameters
    ----------
    optimizer_type : str
        Optimizer type, i.e. PyTorch name of the optimizer
    model : torch.nn.Module
        Neural network model
    learning_rate : float
        Learning rate

    Returns
    -------
    torch.optim.Optimizer
        Instance of optimizer_type optimizer
    """
    optimizer = None
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    else:
        print(f"Unknow optimizer type: {optimizer_type}")
        
    return optimizer


def train_one_epoch(model, data_loader, loss_function, optimizer, device=None, debug=False):

    # Enable training
    model.train(True)

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):
        inputs, labels = batch

        # Move batch to GPU if required
        # if device != torch.device("cpu"):
        #     batch = [i.to(device) for i in batch]

        # Zero gradient
        optimizer.zero_grad()

        # Prepare batch
        # batch, erased = prep_data(batch, model.K, device)

        # Make predictions for batch
        outputs = model(inputs)

        # Update accuracy variables
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        batch_correct = (predicted == labels).sum().item()
        correct += batch_correct

        # Compute loss
        loss = loss_function(outputs, labels)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            # Batch loss
            print(f"    Batch {i}: accuracy={batch_correct / labels.size(0)} | loss={loss}")

    # Compute validation accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss


def evaluate(model, data_loader, loss_function, device, debug=False):

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
            inputs, labels = batch
            
            # Move batch to GPU if required
            # if device != torch.device("cpu"):
            #     batch = [i.to(device) for i in batch]
            
            # Prepare batch
            # batch, erased = prep_data(batch, model.K, device)

            # Make predictions for batch
            outputs = model(inputs)

            # Update accuracy variables
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct = (predicted == labels).sum().item()

            # Compute loss
            loss = loss_function(outputs, labels)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    validation_accuracy = correct / total
    validation_loss /= (i + 1) # Average loss over all batches of the epoch

    return validation_accuracy, validation_loss


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
        debug=False):

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
                    train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, device, debug)
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
            else:
                train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, device, debug)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            # Evaluate model
            validation_accuracy, validation_loss = evaluate(model, validation_data_loader, loss_function, device, debug)
            validation_accuracies.append(validation_accuracy)
            validation_losses.append(validation_loss)

            print(f"Train: accuracy={train_accuracy} | loss={train_loss}")
            print(f"Validation: accuracy={validation_accuracy} | loss={validation_loss}")

            # Early stopping
            if early_stopping:
                if epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta:
                    counter += 1
                    if counter >= patience:
                        print("==== Early Stopping ====")
                        break
                else:
                    counter = 0

        run_epochs.append(epoch + 1)

    return train_accuracies, train_losses, validation_accuracies, validation_losses, run_epochs


def test(model, test_dataloader, device, debug):
    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_labels = None
    all_predicted = None

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            inputs, labels = batch

            # Move batch to GPU
            # if device != torch.device("cpu"):
            #     batch = [i.to(device) for i in batch]

            # Prepare batch
            # batch, erased = prep_data(batch, model.K, device)
            # labels = batch[5]
            
            # Make predictions for batch
            outputs = model(inputs)

            # Update accuracy variables
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

            # Update confusion matrix variables
            if all_labels is None and all_predicted is None:
                all_labels = labels.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_labels = torch.cat((all_labels, labels))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())

    return test_accuracy, test_confusion_matrix


def plot_results(train_accuracies, train_losses,
                 validation_accuracies, validation_losses, run_epochs,
                 architecture, start_timestamp, device,
                 loss_function, optimizer_type, epochs,
                 learning_rates, early_stopping, patience, min_delta,
                 test_accuracy, test_confusion_matrix,
                 stop_timestamp, model_path):
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    # Compute total number of run epochs
    nb_epochs = sum(run_epochs)
    t = np.arange(nb_epochs)

    # Plot accuracy over time
    axs[0, 0].plot(t, train_accuracies, label="Train accuracy")
    axs[0, 0].plot(t, validation_accuracies, label="Validation accuracy")
    offset = 0
    for e in run_epochs:
        axs[0, 0].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0, 0].legend()

    # Plot loss over time
    axs[0, 1].plot(t, train_losses, label="Train loss")
    axs[0, 1].plot(t, validation_losses, label="Validation loss")
    offset = 0
    for e in run_epochs:
        axs[0, 1].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0, 1].legend()

    # Plot confusion matrix
    sns.heatmap(test_confusion_matrix, annot=True, cmap="flare",  fmt="d", cbar=True, ax=axs[1, 0])

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
        ["Test accuracy", test_accuracy],
        ["Stop training", stop_timestamp.strftime("%Y/%m/%d %H:%M:%S")]
    ]

    axs[1, 1].table(cellText=data, loc="center")
    axs[1, 1].axis("off")

    plt.savefig(model_path + ".png")