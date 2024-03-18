import sys
sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")
from src.setup import setup_python, setup_pytorch
import random
from copy import deepcopy
import itertools
import torch
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.combined_model.train_contrastive import contrastive_loss
from src.train import create_optimizer
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as TF
from torchvision.transforms.functional import to_pil_image
import torchvision

#Is used to create the transformed image for input 
def image_augmentation(img, target_size):
    random_resized_crop = transforms.RandomResizedCrop(target_size)
    crop_resized_img = random_resized_crop.forward(img)
    color_jitter = transforms.ColorJitter(0.2,0.2,0.2,0.2)
    augmented_img = color_jitter.forward(crop_resized_img)
    
    return augmented_img
   
def train_one_epoch(model, data_loader, optimizer, device):

    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0
    
    # Pass over all batches
    for i, batch in enumerate(data_loader):

        # Load and prepare batch
        image1, image2 = batch

        img1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = image1
        img2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = image2


        # Move images to device
        img1 = img1.to(device)
        img2 = img2.to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Forward pass
        encoded_img1, decoded_img1 = model(img1)
        
        print("decoded img", decoded_img1.shape)
        print("img", img1.shape)
        #Compute_loss
        loss = TF.mse_loss(decoded_img1,img1)

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
        i += 1

    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_loss


def train(
        model,
        train_data_loader,
        optimizer_type,
        epochs_list,
        learning_rates_list,
        device=torch.device("cpu"),
        debug=False):



    # Losses
    train_losses = []


    for epochs, learning_rate in list(zip(epochs_list, learning_rates_list)):

        # Create optimizer
        optimizer = create_optimizer(optimizer_type, model, learning_rate)

        for epoch in range(epochs):
            print(f"#### EPOCH {epoch} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    _, train_loss = train_one_epoch(model, train_data_loader, optimizer, device)
                    train_losses.append(train_loss)
            else:
                _, train_loss = train_one_epoch(model, train_data_loader, optimizer, device)
                train_losses.append(train_loss)


    return train_losses

