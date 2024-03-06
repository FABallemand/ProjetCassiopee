import torch 
import torch.nn as nn
import torch.nn.functional as F


class TestCNN(nn.Module):
    def __init__(self, input_size, nb_classes=10):
        super().__init__()
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.conv1 = nn.Conv2d(3, 6, 5)  # [(input_width - 5 + 2 * 0) / 1 + 1, (input_height - 5 + 2 * 0) / 1 + 1, 6]
        self.pool = nn.MaxPool2d(2, 2)   # [(input_width - 5) / 1 + 1, (input_height - 5) / 1 + 1, 6]
        self.conv2 = nn.Conv2d(6, 16, 5) # [(input_width - 5 + 2 * 0) / 1 + 1, (input_height - 5 + 2 * 0) / 1 + 1, 16]
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    # class TestCNN(nn.Module):
#     def __init__(self, nb_classes=10):
#         super(TestCNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(160*120*32, nb_classes)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out