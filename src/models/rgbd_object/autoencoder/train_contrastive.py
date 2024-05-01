import os
import psutil
import logging
import torch
from sklearn.manifold import TSNE

from ....train import create_optimizer
from ....loss.contrastive_loss import contrastive_reconstruction_loss
from ....plot.reconstruction_plot import reconstruction_plot
    

def train_one_epoch(
        model,
        data_loader,
        reconstruction_loss_function,
        optimizer,
        epoch,
        device,
        results_dir,
        debug=False):

    # Enable training
    model.train(True)

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
        p_encoded_x_1, p_decoded_x_1 = model(p_rgb_1)
        p_encoded_x_2, p_decoded_x_2 = model(p_rgb_2)
        n_encoded_x, n_decoded_x = model(n_rgb)

        # Compute loss
        loss = contrastive_reconstruction_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                               p_decoded_x_1, p_rgb_1, reconstruction_loss_function)

        if loss.isnan().any():
            logging.debug("loss contains NaN")

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            # Batch loss
            logging.info(f"    Batch {i:8}/{len(data_loader)}: loss={loss:.4f}")

            # Print memory usage
            logging.debug(f"        RAM: {psutil.virtual_memory()[2]} % | {psutil.virtual_memory()[3] / 1000000000} GB")
            logging.debug(f"        VRAM: {torch.cuda.memory_allocated() / 1000000000} / {torch.cuda.max_memory_allocated() / 1000000000}")

        # Save model and plot reconstruction
        if i % 100 == 0 and i != 0:
            torch.save(model.state_dict(), os.path.join(results_dir, f"weights_epoch_{epoch}_batch_{i}"))

            # Plot reconstruction
            reconstruction_plot(p_rgb_1.detach().cpu(),
                                p_decoded_x_1.detach().cpu(),
                                p_rgb_1.shape[0],
                                os.path.join(results_dir, f"reconstruction_epoch_{epoch}_batch_{i}"))

        # Manually delete data
        del p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1
        del p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2
        del n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label
        del p_data_1, p_data_2, n_data
        del batch
        del p_encoded_x_1, p_decoded_x_1
        del p_encoded_x_2, p_decoded_x_2
        del n_encoded_x, n_decoded_x
        del loss

        # Clear cache
        if device != torch.device("cpu"):
            logging.debug("Clear GPU cache")
            torch.cuda.empty_cache()

    # Compute training loss
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_loss


def evaluate(
        model,
        data_loader,
        reconstruction_loss_function,
        epoch,
        device,
        results_dir):

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
            p_encoded_x_1, p_decoded_x_1 = model(p_rgb_1)
            p_encoded_x_2, p_decoded_x_2 = model(p_rgb_2)
            n_encoded_x, n_decoded_x = model(n_rgb)

            # Compute loss
            loss = contrastive_reconstruction_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                                   p_decoded_x_1, p_rgb_1, reconstruction_loss_function)

            # Update batch loss
            validation_loss += loss.item()

            # Plot reconstruction
            if i == 0:
                reconstruction_plot(p_rgb_1.detach().cpu(),
                                    p_decoded_x_1.detach().cpu(),
                                    p_rgb_1.shape[0],
                                    os.path.join(results_dir, f"reconstruction_validation_epoch_{epoch}"))

            # Manually delete data
            del p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1
            del p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2
            del n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label
            del p_data_1, p_data_2, n_data
            del batch
            del p_encoded_x_1, p_decoded_x_1
            del p_encoded_x_2, p_decoded_x_2
            del n_encoded_x, n_decoded_x
            del loss

    # Compute validation loss
    validation_loss /= (i + 1) # Average loss over all batches

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
        device=torch.device("cpu"),
        results_dir="test",
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

        for epoch in range(1, epochs + 1):
            logging.info(f"#### EPOCH {epoch:4}/{epochs} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, epoch, device, results_dir, debug)
                    train_losses.append(train_loss)
                    
                    # Print memory usage
                    # logging.debug(f"RAM: {psutil.virtual_memory()[2]} % | {psutil.virtual_memory()[3] / 1000000000} GB")
                    # logging.debug(f"VRAM: {torch.cuda.memory_allocated() / 1000000000} / {torch.cuda.max_memory_allocated() / 1000000000}")

                    # Print gradients for each parameter
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         logging.debug(f'{name}.grad: mean={param.grad.mean()} | std={param.grad.std()}')
            else:
                train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, epoch, device, results_dir)
                train_losses.append(train_loss)

            # Save model
            torch.save(model.state_dict(), os.path.join(results_dir, f"weights_epoch_{epoch}"))

            # Clear cache
            if device != torch.device("cpu"):
                logging.debug("Clear GPU cache")
                torch.cuda.empty_cache()

            # Evaluate model
            validation_loss = evaluate(model, validation_data_loader, loss_function, epoch, device, results_dir)
            validation_losses.append(validation_loss)

            # Clear cache
            if device != torch.device("cpu"):
                logging.debug("Clear GPU cache")
                torch.cuda.empty_cache()
                
            logging.info(f"Train:      loss={train_loss:.8f}")
            logging.info(f"Validation: loss={validation_loss:.8f}")

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

    return train_losses, validation_losses, run_epochs


def test(model, test_data_loader, tsne_flag=True, device=torch.device("cpu")):

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

            # Save encoded features and labels
            encoded_features.append(p_encoded_x_1)
            labels.append(p_label_1)
            encoded_features.append(p_encoded_x_2)
            labels.append(p_label_2)
            encoded_features.append(n_encoded_x)
            labels.append(n_label)

    # TSNE
    if tsne_flag:
        # Compute batch size
        batch_size = p_rgb_1.shape[0]

        # Compute number of samples
        nb_samples = 3 * (i + 1) * batch_size

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

    return tsne_results_2d, tsne_results_3d, labels_arr