import os
import logging
import torch

from ....train import create_optimizer


def train_one_epoch(model, data_loader, loss_function, optimizer, epoch, results_dir, device):

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
        rgb, depth, mask, loc_x, loc_y, label = batch
        rgb = rgb.to(device)
        # depth = depth.to(device)
        # mask = mask.to(device)
        # loc_x = loc_x.to(device)
        # loc_y = loc_y.to(device)
        label = label.to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        output = model(rgb)

        # Update accuracy variables
        _, predicted = torch.max(output.data, 1)
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
        if i % 10 == 0:
            # Batch loss
            logging.info(f"    Batch {i:8}/{len(data_loader)}: accuracy={batch_correct / label.size(0):.4f} | loss={loss:.4f}")

        # Save model
        if i % 1000 == 0 and i != 0:
            torch.save(model.state_dict(), os.path.join(results_dir, f"weights_epoch_{epoch}_batch_{i}"))

    # Compute train accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss


def evaluate(model, data_loader, loss_function, device):

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
            rgb, depth, mask, loc_x, loc_y, label = batch
            rgb = rgb.to(device)
            # depth = depth.to(device)
            # mask = mask.to(device)
            # loc_x = loc_x.to(device)
            # loc_y = loc_y.to(device)
            label = label.to(device)

            # Make predictions for batch
            output = model(rgb)

            # Update accuracy variables
            _, predicted = torch.max(output.data, 1)
            total += len(label)
            correct = (predicted == label).sum().item()

            # Compute loss
            loss = loss_function(output, label)

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
        results_dir="test",
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
        optimizer = create_optimizer(optimizer_type, model, learning_rate, momentum=0.6)

        for epoch in range(1, epochs + 1):
            logging.info(f"#### EPOCH {epoch:4}/{epochs} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, epoch, results_dir, device)
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)

                    # Print gradients for each parameter
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            logging.debug(f'{name}.grad: mean={param.grad.mean()} | std={param.grad.std()}')
            else:
                train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, epoch, results_dir, device)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            # Save model
            torch.save(model.state_dict(), os.path.join(results_dir, f"weights_epoch_{epoch}"))

            # Evaluate model
            validation_accuracy, validation_loss = evaluate(model, validation_data_loader, loss_function, device)
            validation_accuracies.append(validation_accuracy)
            validation_losses.append(validation_loss)

            logging.info(f"Train:      accuracy={train_accuracy:.8f}      | loss={train_loss:.8f}")
            logging.info(f"Validation: accuracy={validation_accuracy:.8f} | loss={validation_loss:.8f}")

            # Early stopping
            if early_stopping:
                if epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta:
                    counter += 1
                    if counter >= patience:
                        logging.info("==== Early Stopping ====")
                        break
                else:
                    counter = 0

        run_epochs.append(epoch + 1)

    return train_accuracies, train_losses, validation_accuracies, validation_losses, run_epochs


def test(model, test_data_loader, device=torch.device("cpu")):
    # Accuracy variables
    correct = 0
    total = 0

    # Variables
    all_label = None
    all_predicted = None

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            rgb, depth, mask, loc_x, loc_y, label = batch
            rgb = rgb.to(device)
            # depth = depth.to(device)
            # mask = mask.to(device)
            # loc_x = loc_x.to(device)
            # loc_y = loc_y.to(device)
            label = label.to(device)
            
            # Make predictions for batch
            output = model(rgb)

            # Update accuracy variables
            _, predicted = torch.max(output.data, 1)
            total += len(label)
            correct += (predicted == label).sum().item()

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = label.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, label))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total
    logging.info(f"Test:       accuracy={test_accuracy:.8f}")

    return all_label.cpu(), all_predicted.cpu()