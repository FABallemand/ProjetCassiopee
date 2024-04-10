import torch
import torch.nn as nn
    
class RNN(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).double()
        
        out, _ = self.rnn(x, h0)
        
        # Reshape output to (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


#--------------------------------#


class RNN_(nn.Module):
    def __init__(self, input_size, batch_size) :
        super().__init__()
        self.rnn1 = nn.RNN(input_size, 32)
        self.rnn2 = nn.RNN(32, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

        self.batch_size = batch_size
    
    def forward(self, x) :
        x = self.rnn1(x)
        x = self.relu(x)
        x = self.rnn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x
    
    def loss(self, x, y) :
        return nn.CrossEntropyLoss()(self.forward(x), y)