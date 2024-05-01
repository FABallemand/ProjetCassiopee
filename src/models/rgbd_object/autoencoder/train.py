import os
import psutil
import logging
import torch
from sklearn.manifold import TSNE

from ....train import create_optimizer
from ....plot.reconstruction_plot import reconstruction_plot


def train_one_epoch(
        model,
        data_loader,
        loss_function,
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
        encoded, decoded = model(rgb)

        if decoded.isnan().any():
            logging.debug("decoded contains NaN")

        # Compute loss
        loss = loss_function(decoded, rgb)

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
            reconstruction_plot(rgb.detach().cpu(),
                                decoded.detach().cpu(),
                                rgb.shape[0],
                                os.path.join(results_dir, f"reconstruction_epoch_{epoch}_batch_{i}"))

        # Manually delete data
        del rgb, depth, mask, loc_x, loc_y, label
        del batch
        del encoded, decoded
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
        loss_function,
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
            rgb, depth, mask, loc_x, loc_y, label = batch
            rgb = rgb.to(device)
            # depth = depth.to(device)
            # mask = mask.to(device)
            # loc_x = loc_x.to(device)
            # loc_y = loc_y.to(device)
            label = label.to(device)

            # Make predictions for batch
            encoded, decoded = model(rgb)

            # Compute loss
            loss = loss_function(decoded, rgb)

            # Update batch loss
            validation_loss += loss.item()
        
            # Plot reconstruction
            if i == 0:
                reconstruction_plot(rgb.detach().cpu(),
                                    decoded.detach().cpu(),
                                    rgb.shape[0],
                                    os.path.join(results_dir, f"reconstruction_validation_{epoch}"))

            # Manually delete data
            del rgb, depth, mask, loc_x, loc_y, label
            del batch
            del encoded, decoded
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

    # Run inference
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
            encoded, decoded = model(rgb)

            # Save encoded features and labels
            encoded_features.append(encoded)
            labels.append(label)

    # TSNE
    if tsne_flag:
        # Compute batch size
        batch_size = rgb.shape[0]

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

    return tsne_results_2d, tsne_results_3d, labels_arr