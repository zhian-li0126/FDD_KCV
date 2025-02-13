import os
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from es_et_cb import Callback  # same callback used for PyTorch models


class Model:
    def __init__(self):
        """
        Initializes the model architecture.
        """
        self.result = {}

    def train_model(self, x, y, n_estimators=100, max_features='log2', min_samples_split=3, min_samples_leaf=1, k_folds=4, repeats=5, model_save_path="data"):
        """
        Trains a RandomForestClassifier using repeated K-Fold Cross-Validation,
        with a callback pattern to save the best model based on validation accuracy.

        Parameters:
        ----------
        x : np.ndarray
            Feature array, shape (num_samples, num_features).

        y : np.ndarray
            Label array, shape (num_samples,).

        n_estimators : int, default=100
            Number of trees in the forest.

        max_features : {'sqrt', 'log2', int, float}, default='sqrt'
            Number of features to consider when looking for the best split.

        min_samples_split : int or float, default=5
            Minimum number of samples required to split an internal node.

        min_samples_leaf : int or float, default=1
            Minimum number of samples required to be at a leaf node.

        k_folds : int, default=4
            Number of splits in K-Fold cross-validation.

        repeats : int, default=5
            How many times to repeat the entire K-Fold procedure.

        model_save_path : str, default="data"
            Directory to save the best model file.

        Returns:
        --------
        result : dict
            Dictionary with fold indices, repeats, and accuracy metrics:
            {
            'fold': [...],
            'repeat': [...],
            'train_accuracy': [...],
            'val_accuracy': [...]
            }
        """
        print("Training Random Forest for EC diagnosis...")
        # Ensure save path exists
        os.makedirs(model_save_path, exist_ok=True)
        best_model_filename = os.path.join(model_save_path, "best_EC_rf_model.pkl")

        # Prepare result tracking
        fold_list = []
        repeat_list = []
        train_acc_list = []
        val_acc_list = []

        result = {
            "fold": fold_list,
            "repeat": repeat_list,
            "train_accuracy": train_acc_list,
            "val_accuracy": val_acc_list
        }

        # Callback to keep track of the best model across all folds/repeats
        callback = Callback()

        # Set up K-Fold cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=64)

        # Repeat the K-Fold procedure multiple times
        for r in range(repeats):
            for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=k_folds, desc=f"Repeat {r+1}/{repeats}"):
                # Split the dataset into train/val sets for this fold
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create & train RandomForest for this fold with the given hyperparams
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42  # fix seed for reproducibility
                )

                model.fit(x_train, y_train)

                # Evaluate on training set
                train_preds = model.predict(x_train)
                train_acc = accuracy_score(y_train, train_preds)

                # Evaluate on validation set
                val_preds = model.predict(x_val)
                val_acc = accuracy_score(y_val, val_preds)

                print(
                    f"Repeat {r+1}, Fold {fold+1}, "
                    f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}"
                )

                # Store metrics
                fold_list.append(fold + 1)
                repeat_list.append(r + 1)
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)

                # Check if this is the best model so far & save if so
                callback.check_and_save_best_model(model, val_acc)

        # After all folds & repeats, save the best model found
        callback.save_best_model(best_model_filename)
        print(f"✅ Training complete. Best RF model saved to: {best_model_filename}")

        return result
