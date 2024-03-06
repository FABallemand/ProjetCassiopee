import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from ...train import create_optimizer

def train_one_epoch(model, data_loader, loss_function, optimizer):

    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):
        rgb, depth, mask, loc_x, loc_y, label = batch

        # Zero gradient
        optimizer.zero_grad()

        # Prepare batch
        # Data preprocessing?

        # Make predictions for batch
        encoded, decoded = model(rgb)

        # Compute loss
        loss = loss_function(decoded, rgb)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            # Batch loss
            print(f"    Batch {i}: loss={loss}")

    # Compute validation accuracy and loss
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_loss


def evaluate(model, data_loader, loss_function):

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            rgb, depth, mask, loc_x, loc_y, label = batch

            # Make predictions for batch
            encoded, decoded = model(rgb)

            # Compute loss
            loss = loss_function(decoded, rgb)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    validation_loss /= (i + 1) # Average loss over all batches of the epoch

    return validation_loss


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
        debug=False):

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
                    train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer)
                    train_losses.append(train_loss)
            else:
                train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer)
                train_losses.append(train_loss)

            # Evaluate model
            validation_accuracy, validation_loss = evaluate(model, validation_data_loader, loss_function)
            validation_losses.append(validation_loss)

            print(f"Train: loss={train_loss}")
            print(f"Validation: loss={validation_loss}")

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

    return train_losses, validation_losses, run_epochs


def test(model, test_dataloader):
    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_labels = None
    all_predicted = None

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            rgb, depth, mask, loc_x, loc_y, label = batch
            
            # Make predictions for batch
            encoded, decoded = model(rgb)
            
    # ???

    return None