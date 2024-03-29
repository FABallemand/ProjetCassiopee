import torch
import torch.nn.functional as TF
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from ....train import create_optimizer
from ....loss.contrastive_loss import contrastive_classification_loss
    

def train_one_epoch(model, data_loader, classification_loss_function, optimizer, device):

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
        p_data_1, p_data_2, n_data = batch
        p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
        p_rgb_1 = p_rgb_1.to(device)
        # p_depth_1 = p_depth_1.to(device)
        # p_mask_1 = p_mask_1.to(device)
        # p_loc_x_1 = p_loc_x_1.to(device)
        # p_loc_y_1 = p_loc_y_1.to(device)
        p_label_1 = p_label_1.to(device)
        p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
        p_rgb_2 = p_rgb_2.to(device)
        # p_depth_2 = p_depth_2.to(device)
        # p_mask_2 = p_mask_2.to(device)
        # p_loc_x_2 = p_loc_x_2.to(device)
        # p_loc_y_2 = p_loc_y_2.to(device)
        p_label_2 = p_label_2.to(device)
        n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
        n_rgb = n_rgb.to(device)
        # n_depth = n_depth.to(device)
        # n_mask = n_mask.to(device)
        # n_loc_x = n_loc_x.to(device)
        # n_loc_y = n_loc_y.to(device)
        n_label = n_label.to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        p_encoded_x_1, p_predicted_label_1 = model(p_rgb_1)
        p_encoded_x_2, p_predicted_label_2 = model(p_rgb_2)
        n_encoded_x, n_predicted_label = model(n_rgb)

        # Update accuracy variables
        _, predicted = torch.max(p_predicted_label_1.data, 1)
        total += len(p_label_1)
        batch_correct = (predicted == p_label_1).sum().item()
        correct += batch_correct

        # Compute loss
        loss = contrastive_classification_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                               p_predicted_label_1, p_label_1, classification_loss_function)

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            # Batch loss
            print(f"    Batch {i}: accuracy={batch_correct / p_label_1.size(0)} | loss={loss}")
        i += 1

    # Compute validation accuracy and loss
    train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss


def evaluate(model, data_loader, classification_loss_function, device):

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
            p_data_1, p_data_2, n_data = batch
            p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
            p_rgb_1 = p_rgb_1.to(device)
            # p_depth_1 = p_depth_1.to(device)
            # p_mask_1 = p_mask_1.to(device)
            # p_loc_x_1 = p_loc_x_1.to(device)
            # p_loc_y_1 = p_loc_y_1.to(device)
            p_label_1 = p_label_1.to(device)
            p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
            p_rgb_2 = p_rgb_2.to(device)
            # p_depth_2 = p_depth_2.to(device)
            # p_mask_2 = p_mask_2.to(device)
            # p_loc_x_2 = p_loc_x_2.to(device)
            # p_loc_y_2 = p_loc_y_2.to(device)
            p_label_2 = p_label_2.to(device)
            n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
            n_rgb = n_rgb.to(device)
            # n_depth = n_depth.to(device)
            # n_mask = n_mask.to(device)
            # n_loc_x = n_loc_x.to(device)
            # n_loc_y = n_loc_y.to(device)
            n_label = n_label.to(device)

            # Make predictions for batch
            p_encoded_x_1, p_predicted_label_1 = model(p_rgb_1)
            p_encoded_x_2, p_predicted_label_2 = model(p_rgb_2)
            n_encoded_x, n_predicted_label = model(n_rgb)

            # Update accuracy variables
            _, predicted = torch.max(p_predicted_label_1.data, 1)
            total += len(p_label_1)
            batch_correct = (predicted == p_label_1).sum().item()
            correct += batch_correct

            # Compute loss
            loss = contrastive_classification_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                                   p_predicted_label_1, p_label_1, classification_loss_function)

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
                    train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, device)
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
            else:
                train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, device)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            # Evaluate model
            validation_accuracy, validation_loss = evaluate(model, validation_data_loader, loss_function, device)
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


def test(model, test_data_loader, tsne_flag=True, device=torch.device("cpu")):
    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_label = None
    all_predicted = None

    # TSNE variable
    encoded_features = []
    labels = []
    tsne_results_2d = None
    tsne_results_3d = None

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            p_data_1, p_data_2, n_data = batch
            p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
            p_rgb_1 = p_rgb_1.to(device)
            # p_depth_1 = p_depth_1.to(device)
            # p_mask_1 = p_mask_1.to(device)
            # p_loc_x_1 = p_loc_x_1.to(device)
            # p_loc_y_1 = p_loc_y_1.to(device)
            p_label_1 = p_label_1.to(device)
            p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
            p_rgb_2 = p_rgb_2.to(device)
            # p_depth_2 = p_depth_2.to(device)
            # p_mask_2 = p_mask_2.to(device)
            # p_loc_x_2 = p_loc_x_2.to(device)
            # p_loc_y_2 = p_loc_y_2.to(device)
            p_label_2 = p_label_2.to(device)
            n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
            n_rgb = n_rgb.to(device)
            # n_depth = n_depth.to(device)
            # n_mask = n_mask.to(device)
            # n_loc_x = n_loc_x.to(device)
            # n_loc_y = n_loc_y.to(device)
            n_label = n_label.to(device)
            
            # Make predictions for batch
            p_encoded_x_1, p_predicted_label_1 = model(p_rgb_1)
            p_encoded_x_2, p_predicted_label_2 = model(p_rgb_2)
            n_encoded_x, n_predicted_label = model(n_rgb)

            # Update accuracy variables
            _, predicted = torch.max(p_predicted_label_1.data, 1)
            total += len(p_label_1)
            batch_correct = (predicted == p_label_1).sum().item()
            correct += batch_correct

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = p_label_1.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, p_label_1))
                all_predicted = torch.cat((all_predicted, predicted))

            # Save encoded features and labels
            encoded_features.append(p_encoded_x_1)
            labels.append(p_label_1)
            encoded_features.append(p_encoded_x_2)
            labels.append(p_label_2)
            encoded_features.append(n_encoded_x)
            labels.append(n_label)
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_label.cpu(), all_predicted.cpu())

    # TSNE
    if tsne_flag:
        # Compute batch size
        batch_size = p_rgb_1.shape[0]

        # Compute number of samples
        nb_samples = (i + 1) * batch_size

        # Process inference results
        encoded_features_arr = torch.empty(size=(nb_samples, 256))
        for i, batch in enumerate(encoded_features):
            encoded_features_arr[i * batch_size:(i + 1) * batch_size,:] = batch
        labels_arr = torch.empty(size=(nb_samples,))
        for i, batch in enumerate(labels):
            labels_arr[i * batch_size:(i + 1) * batch_size] = batch

        # Apply 2D TSNE
        tsne_2d = TSNE(n_components=2,
                       perplexity=30,
                       n_iter=1000,
                       init="pca",
                       random_state=42)
        tsne_results_2d = tsne_2d.fit_transform(encoded_features_arr)

        # Apply 3D TSNE
        tsne_3d = TSNE(n_components=3,
                       perplexity=30,
                       n_iter=1000,
                       init="pca",
                       random_state=42)
        tsne_results_3d = tsne_3d.fit_transform(encoded_features_arr)

    return test_accuracy, test_confusion_matrix, tsne_results_2d, tsne_results_3d, labels_arr