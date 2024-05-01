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

from rnn import RNN_
from rnn import RNN 
from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatasetRNN

#intialize the RNN model
rnn = RNN_(input_size=237, softmax_activated=False).double()

# set the evaluation mode
rnn.eval()

# get the image form the dataloader
dataset = MocaplabDatasetRNN(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)

img = next(iter(data_loader))[0]
#img_flattened = img.view(img.size(0), -1)
#img_flattened = img_flattened.unsqueeze(1)

# get the most likely prediction of the model
pred = rnn(img.double())

# get the gradient of the output with respect to the parameters of the model
pred[:,0].backward()

# pull the gradients out of the model
gradients = rnn.get_activations_gradient()


# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2])

# get the activations of the last convolutional layer
activations = rnn.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(256):
    activations[:, i, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1)

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

fig = plt.figure()
# draw the heatmap
plt.matshow(heatmap, aspect='auto')
plt.savefig("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/heatmap_rnn.png")
plt.close()

#To superimpose the heatmap on the array
import cv2

array = cv2.imread("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/array/array.png")
heatmap = cv2.imread("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/heatmap_rnn.png")
heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + array
cv2.imwrite('/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/heatmap/map_rnn.png', superimposed_img)
