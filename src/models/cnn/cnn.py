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

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, "same")    # Convolution     -> [128, 128, 64]
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, "same")   # Convolution     -> [128, 128, 64]
                                                         # Pooling         -> [64, 64, 64]
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, "same")  # Convolution     -> [64, 64, 128]
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, "same") # Convolution     -> [64, 64, 128]
                                                         # Pooling         -> [32, 32, 128]
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, "same") # Convolution     -> [32, 32, 256]
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, "same") # Convolution     -> [32, 32, 256]
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, "same") # Convolution     -> [32, 32, 256]
                                                         # Pooling         -> [16, 16, 256]
        self.fc1 = nn.Linear(16 * 16 * 256, 1024)        # Fully connected -> [1024]
        self.fc2 = nn.Linear(1024, 512)                  # Fully connected -> [512]
        self.fc3 = nn.Linear(512, self.nb_classes)       # Fully connected -> [self.nb_classes]

    def forward(self, x):
        
        # print("Conv 1")
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        # print("Pool 1")
        x = self.pool(x)

        # print("Conv 2")
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        # print("Pool 2")
        x = self.pool(x)

        # print("Conv 3")
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))       
        x = F.relu(self.conv3_3(x))

        # print("Pool 3")
        x = self.pool(x)

        # print("Flatten")
        x = torch.flatten(x, 1)

        # print("Fully Connected 1")
        x = F.relu(self.fc1(x))

        # print("Fully Connected 2")
        x = F.relu(self.fc2(x))

        # print("Fully Connected 3")
        x = F.softmax(self.fc3(x), dim=1)

        # print("Output")
        return x