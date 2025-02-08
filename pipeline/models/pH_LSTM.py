import torch
import torch.nn as nn
import torch.optim as optim

class LSTMDenseModel(nn.Module):
    def __init__(self):
        super(LSTMDenseModel, self).__init__()
        
        self.fc1 = nn.Linear(in_features=7, out_features=64)  # Adjust input features based on your dataset
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=6, batch_first=True)  # LSTM with 6 hidden units
        self.fc4 = nn.Linear(6, 6)  # Final output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        x = x.view(x.shape[0], 64, 1)  # Reshape to (batch_size, 64, 1) for LSTM
        
        x, _ = self.lstm(x)  # LSTM processing
        x = x[:, -1, :]  # Take the last time step's output
        
        x = self.fc4(x)  # Fully connected output
        return x