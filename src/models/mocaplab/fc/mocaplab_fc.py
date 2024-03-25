import torch
import torch.nn as nn

class MocaplabFC(nn.Module):
    def __init__(self, input_size) :
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x) :
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x
    
    def loss(self, x, y) :
        return nn.CrossEntropyLoss()(self.forward(x), y)