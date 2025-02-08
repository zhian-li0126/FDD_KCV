import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Model(nn.Module):
    '''
    Parameters:
    Class:
    Description:
    '''
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(in_features=7, out_features=64)  # Adjust input features based on your dataset
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.lstm = nn.LSTM(input_size=1, hidden_size=6, batch_first=True)  # LSTM with 6 hidden units
        self.fc4 = nn.Linear(6, 6)  # Final output layer
        
        self.result = {}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        x = x.view(x.shape[0], 64, 1)  # Reshape to (batch_size, 64, 1) for LSTM
        
        x, _ = self.lstm(x)  # LSTM processing
        x = x[:, -1, :]  # Take the last time step's output
        
        x = self.fc4(x)  # Fully connected output
        return x

    def train(self, x, y, num_epoch = 50, k_folds = 4, repeats = 5, lr = 1e-3) -> dict:
        '''
        parameters:
        description:
        '''
        # Initialize the saved metrics in KCV
        train_acc = []
        train_loss = []
        val_loss = []
        val_acc = []
        repeat = []
        fold = []
        result = {"fold": fold, "repeat": repeat, 
                  "training loss": train_loss, "training accuracy":train_acc, 
                  "validation loss": val_loss, "validation accuracy":val_acc}

        
        # Define KCV parameters
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        dataset = x  # Assign dataset for splitting

        for repeat in range(repeats):  # Ensure repeats is iterated correctly
            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                # Initialized running loss and accuracy
                average_running_loss = 0
                average_running_acc = 0
                
                for epoch in num_epoch:
                    print(f"Repeat {repeat + 1}, Fold {fold + 1}")
                    x_train, x_val = dataset[train_idx], dataset[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    optimizer.zero_grad()
                    outputs = self(x_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        
                        outputs = self(x_val)
                        loss = criterion(x_val, y_val)
                        
                        running_test_loss += loss.item()
                        running_test_acc += accuracy_score(x_val, y_val)
                    
                    average_running_loss = running_test_loss / len(kf.split(dataset))
                    average_running_acc = running_test_acc / len(kf.split(dataset))
                    val_loss.append(average_running_loss)
                    val_acc.append(average_running_acc)


        print("✅ Training complete for pH_LSTM.")
        
        return result
