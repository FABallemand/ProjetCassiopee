import torch
import torch.nn as nn
from autoencoder import Autoencoder
from cnn import TestCNN

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel,self).__init__()
        #Initialize autoencoder and encoder
        self.autoencoder = Autoencoder()
        self.encoder = TestCNN()

    def forward(self, x):
        #Forward pass through autoencoder and encoder
        encoded_output, reconstructed_output = self.autoencoder(x)
        output = self.encoder(reconstructed_output)
        return encoded_output, output

