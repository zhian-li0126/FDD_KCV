import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from es_et_cb import EarlyStopping, Callback
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os

class Model(nn.Module):
    """
    A feedforward neural network with an NLP layer for classification tasks.

    Architecture:
    - 3 fully connected layers with ReLU activation.
    - LSTM with 6 hidden units.
    - Final fully connected output layer.

    Methods:
    - forward: Forward pass through the network.
    - train_model: Trains the model using k-fold cross-validation, 
                   early stopping, and best model saving.
    """

    def __init__(self):
        """
        Initializes the model architecture.
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=7, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.result = {}

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, 6).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, 6).
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Final fully connected output
        return x

    def train_model(self, x, y, num_epoch=1000, k_folds=4, repeats=5, lr=1e-3, patience=50, model_save_path="data", batch_size=64):
        """
        Trains the model using k-fold cross-validation with early stopping 
        and saves the best model based on validation accuracy.

        Parameters:
        - x (np.ndarray): Feature array, shape (num_samples, num_features).
        - y (np.ndarray): Label array, shape (num_samples,).
        - num_epoch (int): Number of epochs for training.
        - k_folds (int): Number of folds in K-Fold Cross-Validation.
        - repeats (int): Number of times to repeat k-fold validation.
        - lr (float): Learning rate for Adam optimizer.
        - patience (int): Patience for early stopping (default: 200).
        - model_save_path (str): Path to save the best model.
        - batch_size (int): Mini-batch size for DataLoader.

        Returns:
        - dict: Training and validation metrics (best per fold).
        """
        print("Training EC ANN model...")
        # Initialize tracking lists
        train_acc = []
        train_loss = []
        val_loss = []
        val_acc = []
        fold_list = []
        repeat_list = []

        result = {
            "fold": fold_list,
            "repeat": repeat_list,
            "training loss": train_loss,
            "training accuracy": train_acc,
            "validation loss": val_loss,
            "validation accuracy": val_acc
        }

        # Set up K-Fold cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=64)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Set up callback for saving best models
        callback = Callback()

        # Ensure model save path exists
        os.makedirs(model_save_path, exist_ok=True)
        best_model_filename = os.path.join(model_save_path, "best_EC_ANN_model.pth")

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Convert x,y to PyTorch tensors on the correct device
        x_tensor = torch.from_numpy(x).float().to(device)
        y_tensor = torch.from_numpy(y).long().to(device)

        for r in range(repeats):
            for fold, (train_idx, val_idx) in enumerate(kf.split(x_tensor)):
                print(f"Repeat {r+1}, Fold {fold+1}")
                # Create fresh early stopper each fold
                early_stopper = EarlyStopping(patience=patience, min_delta=0.01)

                # Create Datasets and DataLoaders for train/val
                x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
                x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]

                train_dataset = TensorDataset(x_train, y_train)
                val_dataset   = TensorDataset(x_val,   y_val)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

                # Track best metrics
                best_train_loss = float("inf")
                best_train_acc = 0
                best_val_loss = float("inf")
                best_val_acc = 0

                for epoch in tqdm(range(num_epoch), desc="Training", unit="epoch"):

                    # ======== TRAIN LOOP ========
                    self.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    for x_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = self(x_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()

                        # Accumulate loss/accuracy for this mini-batch
                        running_loss += loss.item() * x_batch.size(0)
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == y_batch).sum().item()
                        total += y_batch.size(0)

                    # Average train loss/acc over all mini-batches
                    epoch_train_loss = running_loss / total
                    epoch_train_acc = correct / total

                    # Track best training metrics
                    if epoch_train_loss < best_train_loss:
                        best_train_loss = epoch_train_loss
                    if epoch_train_acc > best_train_acc:
                        best_train_acc = epoch_train_acc

                    # ======== VALIDATION LOOP ========
                    self.eval()
                    val_running_loss = 0.0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for x_valb, y_valb in val_loader:
                            val_outputs = self(x_valb)
                            v_loss = criterion(val_outputs, y_valb)
                            val_running_loss += v_loss.item() * x_valb.size(0)
                            val_preds = torch.argmax(val_outputs, dim=1)
                            val_correct += (val_preds == y_valb).sum().item()
                            val_total += y_valb.size(0)

                    # Average val loss/acc over the entire val set
                    epoch_val_loss = val_running_loss / val_total
                    epoch_val_acc = val_correct / val_total

                    # Track best validation metrics
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                    if epoch_val_acc > best_val_acc:
                        best_val_acc = epoch_val_acc

                    # Early stopping check (uses validation loss)
                    if early_stopper.early_stop(epoch_val_loss):
                        print("🛑 Early stopping triggered.")
                        break

                # Store best metrics for this fold
                train_loss.append(best_train_loss)
                train_acc.append(best_train_acc)
                val_loss.append(best_val_loss)
                val_acc.append(best_val_acc)
                fold_list.append(fold + 1)
                repeat_list.append(r + 1)

                # Check if this fold’s best model is the overall best
                callback.check_and_save_best_model(self, best_val_acc)

        # After all folds and repeats, save the best model found
        callback.save_best_model(best_model_filename)

        print(f"✅ Training complete. Best model saved to: {best_model_filename}")
        return result