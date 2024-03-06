import torch
import torch.nn as nn


class Autoencoder(nn.Module) :
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Define encoder layers
        # Define convolutional layers
        self.en_conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.en_conv2 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.en_conv3 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.en_conv4 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv5 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        # Define max pooling layer
        self.maxpool = nn.Maxpool2d(kernel_size=2, stride=2)

        # Define decoder layers
        # Define convolutional layers
        self.de_transconv1 = nn.ConvTranspose2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.de_transconv2 = nn.ConvTranspose2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.de_transconv3 = nn.ConvTranspose2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.de_transconv4 = nn.ConvTranspose2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.de_transconv5 = nn.ConvTranspose2d(3, 16, kernel_size=7, stride=2, padding=3)
        # Define upsampling operation
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Define batch normalisation
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        # Convolutional layer 1
        x = self.en_conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Convolutional layer 2
        x = self.en_conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Convolutional layer 3
        x = self.en_conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Convolutional layer 4
        x = self.en_conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Convolutional layer 5
        x = self.en_conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        encoded = self.maxpool(x)

        # Decoder
        # Transposed convolutional layer 1
        x = self.upsample(encoded)
        x = self.de_transconv1(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        # Transposed convolutional layer 2
        x = self.upsample(encoded)
        x = self.de_transconv2(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        # Transposed convolutional layer 3
        x = self.upsample(encoded)
        x = self.de_transconv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        # Transposed convolutional layer 4
        x = self.upsample(encoded)
        x = self.de_transconv4(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        # Transposed convolutional layer 5
        x = self.upsample(encoded)
        x = self.de_transconv5(x)
        x = self.batchnorm1(x)
        decoded = self.sigmoid(x)

        return encoded, decoded