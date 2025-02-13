import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=7, out_features=64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.lstm = nn.LSTM(input_size=1, hidden_size=6, batch_first=True)
        self.fc4 = nn.Linear(6, 6)
        self.result = {}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x.view(x.shape[0], 64, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc4(x)
        return x

    def train_model(self, x, y, num_epoch=50, k_folds=4, repeats=5, lr=1e-3):
        train_acc = []
        train_loss = []
        val_loss = []
        val_acc = []
        fold_list = []
        repeat_list = []

        result = {"fold": fold_list, "repeat": repeat_list, 
                  "training loss": train_loss, "training accuracy": train_acc, 
                  "validation loss": val_loss, "validation accuracy": val_acc}

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        x = x.to(device)
        y = y.to(device)

        for r in range(repeats):
            for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
                running_train_loss = 0
                running_train_acc = 0
                running_val_loss = 0
                running_val_acc = 0

                for epoch in range(num_epoch):
                    print(f"Repeat {r + 1}, Fold {fold + 1}, Epoch {epoch + 1}")

                    x_train, x_val = x[train_idx], x[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    x_train, x_val = x_train.to(device), x_val.to(device)
                    y_train, y_val = y_train.to(device), y_val.to(device)

                    optimizer.zero_grad()
                    outputs = self(x_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                    preds_train = torch.argmax(outputs, dim=1)
                    running_train_acc += accuracy_score(y_train.cpu().numpy(), preds_train.cpu().numpy())

                    with torch.no_grad():
                        val_outputs = self(x_val)
                        val_loss_value = criterion(val_outputs, y_val).item()
                        preds_val = torch.argmax(val_outputs, dim=1)
                        val_acc_value = accuracy_score(y_val.cpu().numpy(), preds_val.cpu().numpy())

                        running_val_loss += val_loss_value
                        running_val_acc += val_acc_value

                avg_train_loss = running_train_loss / num_epoch
                avg_train_acc = running_train_acc / num_epoch
                avg_val_loss = running_val_loss / num_epoch
                avg_val_acc = running_val_acc / num_epoch

                train_loss.append(avg_train_loss)
                train_acc.append(avg_train_acc)
                val_loss.append(avg_val_loss)
                val_acc.append(avg_val_acc)
                fold_list.append(fold + 1)
                repeat_list.append(r + 1)

        print("✅ Training complete.")
        return result
