import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")
from src.setup import setup_python, setup_pytorch
from src import plot_results
from src.dataset import RGBDObjectDataset, RGBDObjectDataset_Contrast
from src.models.combined_model import CombinedModel, train, test

if __name__=='__main__':

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters
    INPUT_SIZE = (128,128)
    TRANSFORMATION = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size=INPUT_SIZE)])
    NB_TRAIN_SAMPLES = None
    NB_VALIDATION_SAMPLES = None
    NB_TEST_SAMPLES = None

    # Testing parameters
    BATCH_SIZE = 128 # Batch size
    MODEL_PATH = "train_results/"
    
    # Datasets
    print("#### Datasets ####")
    
    test_dataset = RGBDObjectDataset_Contrast(path="data/RGB-D_Object/rgbd-dataset",
                                              mode="test",
                                              transformation=TRANSFORMATION,
                                              nb_samples=NB_TEST_SAMPLES)
    NB_TEST_SAMPLES = len(test_dataset)
    
    print(f"Test dataset -> {len(test_dataset)} samples")
    
    # Data loaders
    print("#### Data Loaders ####")
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True)
    
    # Create neural network
    print("#### Model ####")

    model = CombinedModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Save training time start
    start_timestamp = datetime.now()

    # Create path for saving things...
    results_path = f"tsne_results/tsne_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Run inference
    encoded_features = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            
            # Load and prepare batch
            p_data_1, p_data_2, n_data = batch
            p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
            p_rgb_1 = p_rgb_1.to(DEVICE)
            # p_depth_1 = p_depth_1.to(DEVICE)
            # p_mask_1 = p_mask_1.to(DEVICE)
            # p_loc_x_1 = p_loc_x_1.to(DEVICE)
            # p_loc_y_1 = p_loc_y_1.to(DEVICE)
            p_label_1 = p_label_1.to(DEVICE)
            # p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
            # p_rgb_2 = p_rgb_2.to(DEVICE)
            # p_depth_2 = p_depth_2.to(DEVICE)
            # p_mask_2 = p_mask_2.to(DEVICE)
            # p_loc_x_2 = p_loc_x_2.to(DEVICE)
            # p_loc_y_2 = p_loc_y_2.to(DEVICE)
            # p_label_2 = p_label_2.to(DEVICE)
            # n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
            # n_rgb = n_rgb.to(DEVICE)
            # n_depth = n_depth.to(DEVICE)
            # n_mask = n_mask.to(DEVICE)
            # n_loc_x = n_loc_x.to(DEVICE)
            # n_loc_y = n_loc_y.to(DEVICE)
            # n_label = n_label.to(DEVICE)
            
            # Make predictions for batch
            encoded_x, predicted_label = model(p_rgb_1)

            # Save encoded features and labels
            encoded_features.append(encoded_x)
            labels.append(p_label_1)

    # Process inference results
    encoded_features_arr = torch.empty(size=(NB_TEST_SAMPLES, 4096))
    for i, batch in enumerate(encoded_features):
        encoded_features_arr[i * BATCH_SIZE:(i + 1) * BATCH_SIZE,:] = batch
    labels_arr = torch.empty(size=(NB_TEST_SAMPLES,))
    for i, batch in enumerate(labels):
        labels_arr[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = batch

    # Do PCA before TSNE??

    # Apply TSNE
    tsne = TSNE(n_components=2,
                perplexity=30,
                n_iter=1000,
                init="pca",
                random_state=42)

    tsne_results = tsne.fit_transform(encoded_features_arr)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(tsne_results[:,0], tsne_results[:,1], c=labels_arr, s=50, alpha=0.8)

    # Save results
    plt.savefig(results_path + ".png")

    # End TSNE
    print("#### End ####")