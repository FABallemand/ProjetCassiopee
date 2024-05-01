import torch
import torch.nn as nn
    
class RNN(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #self.fc1 = nn.Linear(input_size, 128)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).double()
        
        #x = self.fc1(x)
        out, _ = self.rnn(x, h0)
        
        # Reshape output to (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        
        # Decode the hidden state of the last time step
        out = self.fc2(out)
        return out


#--------------------------------#


class RNN_(nn.Module):
    def __init__(self, input_size, softmax_activated=True) :
        super().__init__()
        self.rnn1 = nn.RNN(input_size, 32, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.softmax_activated = softmax_activated
        self.gradients = None
    
    def forward(self, x) :

        h1 = torch.zeros(1, x.size(0), 32).to(x.device).double()
        h2 = torch.zeros(1, x.size(0), 64).to(x.device).double()

        x, _ = self.rnn1(x, h1)
        x = self.relu(x)
        x, hn = self.rnn2(x, h2)
        x = self.relu(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)

        if self.softmax_activated :
            x = self.softmax(x)

        return x
    
    def loss(self, x, y) :
        return nn.CrossEntropyLoss()(self.forward(x), y)