import torch 
import torch.nn as nn
import torch.nn.functional as F


class TestCNN(nn.Module):
    """
    Convolutional neural network inspired of VGGNet-16
    Inspired by: https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide
    """
    """
    Uses https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    for heatmap
    """

    def __init__(self, softmax=True, nb_classes=2):
        super().__init__()
        self.nb_classes = nb_classes
        self.softmax = softmax

        self.pool = nn.MaxPool2d(2, 2) # [(input_width - 2) / 2 + 1, (input_height - 2) / 2 + 1, input_depth]

        self.conv1_1 = nn.Conv2d(1, 32, 3, 1, "same")    # Convolution     -> [256, 256, 32]
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, "same")   # Convolution     -> [256, 256, 32]
                                                         # Pooling         -> [128, 128, 32]
        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, "same")   # Convolution     -> [128, 128, 64]
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, "same")   # Convolution     -> [128, 128, 64]
                                                         # Pooling         -> [64, 64, 64]
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, "same")  # Convolution     -> [64, 64, 128]
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, "same") # Convolution     -> [64, 64, 128]
        self.conv3_3 = nn.Conv2d(128, 256, 3, 1, "same") # Convolution     -> [64, 64, 256]
                                                         # Pooling         -> [32, 32, 256]
        self.fc1 = nn.Linear(262144, 1024)                # Fully connected -> [1024]
        self.fc2 = nn.Linear(1024, 512)                  # Fully connected -> [512]
        self.fc3 = nn.Linear(512, self.nb_classes)       # Fully connected -> [self.nb_classes]

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        
        # print("Conv 1")
        x = self.conv1_1(x)
        x = F.relu(x)
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

        #register the hook
        hook = x.register_hook(self.activations_hook)

        # print("Pool 3")
        x = self.pool(x)

        # print("Flatten")
        x = torch.flatten(x, 1)

        # print("Fully Connected 1")
        x = F.relu(self.fc1(x))

        # print("Fully Connected 2")
        x = F.relu(self.fc2(x))
        
        if self.softmax :
            # print("Fully Connected 3")
            x = F.softmax(self.fc3(x), dim=1)
        else :
            x = self.fc3(x)
            
        # print("Output")
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        # print("Conv 1")
        x = self.conv1_1(x)
        x = F.relu(x)
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

        return x