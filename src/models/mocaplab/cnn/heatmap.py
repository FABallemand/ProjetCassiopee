import os
import sys
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")

from cnn import TestCNN
from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatasetCNN


#intialize the CNN model
cnn = TestCNN(softmax=False)

# set the evaluation mode
cnn.eval()

# get the image form the dataloader
dataset = MocaplabDatasetCNN(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)

for i in range(len(data_loader)) :
    img = next(iter(data_loader))[i]

    # get the most likely prediction of the model
    pred = cnn(img)

    # get the gradient of the output with respect to the parameters of the model
    pred[:,0].backward()

    # pull the gradients out of the model
    gradients = cnn.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])


    # get the activations of the last convolutional layer
    activations = cnn.get_activations(img).detach()


    # weight the channels by corresponding gradients
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    #For each time frame, get the coordinates of the ten maximum activation values to find the most significant joints
    #First, reshape the heatmap (64x64) to original size (100x237)
    heatmap_resized = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap_resized,size=(100,237), mode='bilinear')
    #heatmap_resized = torch.squeeze(heatmap_resized)
    ten_max_joints_all_frames = []

    for i in range(1, 101):
        max_for_one_joint = []
        for j in range(0, 79): #237/3
            max = torch.max(heatmap_resized[i][j*3:j*3+2])
            max_for_one_joint.append(max)
        _, max_activations_indices = torch.topk(max_for_one_joint, k=10)
        ten_max_joints_all_frames.append(max_activations_indices)
    print(ten_max_joints_all_frames)

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    nom = str(i)
    fig = plt.figure()
    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.savefig(f"/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/heatmap_cnn2D_{nom}.png")
    plt.close()

#To superimpose the heatmap on the array
'''
import cv2

array = cv2.imread("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/array/array.png")
heatmap = cv2.imread("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/heatmap_cnn2D.png")
heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + array
cv2.imwrite('/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/map_cnn2D.png', superimposed_img)
'''

