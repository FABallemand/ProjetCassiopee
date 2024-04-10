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

img = next(iter(data_loader))[0]


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

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

fig = plt.figure()
# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.savefig("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/heatmap.png")
plt.close()





