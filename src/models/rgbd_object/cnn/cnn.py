import torch 
import torch.nn as nn
import torch.nn.functional as F


class TestCNN(nn.Module):
    """
    Convolutional neural network inspired of VGGNet-16
    Inspired by: https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide
    """
    def __init__(self, nb_classes=10):
        super().__init__()
        self.nb_classes = nb_classes

        self.pool = nn.MaxPool2d(2, 2) # [(input_width - 2) / 2 + 1, (input_height - 2) / 2 + 1, input_depth]

        self.conv1_1 = nn.Conv2d(3, 16, 3, 1, "same")    # Convolution     -> [256, 256, 16]
        self.conv1_2 = nn.Conv2d(16, 32, 3, 1, "same")   # Convolution     -> [256, 256, 32]
                                                         # Pooling         -> [128, 128, 32]
        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, "same")   # Convolution     -> [128, 128, 64]
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, "same")   # Convolution     -> [128, 128, 64]
                                                         # Pooling         -> [64, 64, 64]
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, "same")  # Convolution     -> [64, 64, 128]
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, "same") # Convolution     -> [64, 64, 128]
        self.conv3_3 = nn.Conv2d(128, 256, 3, 1, "same") # Convolution     -> [64, 64, 256]
                                                         # Pooling         -> [32, 32, 256]
        self.fc1 = nn.Linear(32 * 32 * 256, 1024)        # Fully connected -> [1024]
        self.fc2 = nn.Linear(1024, 512)                  # Fully connected -> [512]
        self.fc3 = nn.Linear(512, self.nb_classes)       # Fully connected -> [self.nb_classes]

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))       
        x = F.relu(self.conv3_3(x))

        x = self.pool(x)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x


class TestSmallerCNN(nn.Module):
    """
    Convolutional neural network
    """
    def __init__(self, nb_classes=10):
        super().__init__()
        self.nb_classes = nb_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 7, 1, "same")     # Convolution     -> [256, 256, 8]
        self.conv2 = nn.Conv2d(8, 16, 5, 1, "same")    # Convolution     -> [256, 256, 16]
        self.conv3 = nn.Conv2d(16, 32, 3, 1, "same")   # Convolution     -> [256, 256, 32]
                                                       # Pooling         -> [128, 128, 32]
        self.conv4 = nn.Conv2d(32, 64, 7, 1, "same")   # Convolution     -> [128, 128, 64]
        self.conv5 = nn.Conv2d(64, 128, 5, 1, "same")  # Convolution     -> [128, 128, 128]
        self.conv6 = nn.Conv2d(128, 128, 3, 1, "same") # Convolution     -> [128, 128, 128]
                                                       # Pooling         -> [64, 64, 128]
        self.fc1 = nn.Linear(524288, 512)              # Fully connected -> [512]
        self.fc2 = nn.Linear(512, 256)                 # Fully connected -> [256]
        self.fc3 = nn.Linear(256, self.nb_classes)     # Fully connected -> [self.nb_classes]

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.pool(x)

        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        encoded = x
        x = F.softmax(self.fc3(x), dim=1)

        return x