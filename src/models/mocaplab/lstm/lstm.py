import torch
import torch.nn as nn
    
class LSTM(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.fc1 = nn.Linear(input_size, 512)
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).double()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).double()
        
        x = self.fc1(x)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Reshape output to (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        
        # Decode the hidden state of the last time step
        out = self.fc2(out)
        return out