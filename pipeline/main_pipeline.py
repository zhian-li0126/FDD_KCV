import importlib
import os
from data_loader import load_data
import pandas as pd

# List of model file names (without .py extension)
model_names = ["pH_LSTM", "pH_ANN", "pH_RF", "EC_LSTM", "EC_ANN", "EC_RF"]

# Path to the models directory
models_dir = "models"

# Training function
def train_model(model_name):
    """
    Dynamically imports a model, loads the dataset, and starts training.
    """
    print(f"\n🚀 Training model: {model_name} ...")

    # Load dataset
    x, y = load_data(model_name)

    # Dynamically import the model module
    model_module = importlib.import_module(f"models.{model_name}")
    
    # Instantiate the model (Assuming each model has a `Model` class)
    model = model_module.Model()

    # Train model (Assume each model has a `train` method)
    result = model.train(x, y)
    
    # Save model
    df = pd.DataFrame(result)
    df.to_csv(f"{model_name}.csv", index=False)

    print(f"✅ Training complete for: {model_name}\n
          f"model saved to {os.getCWD()} + {model_name}.csv\n")

# Run all models sequentially
if __name__ == "__main__":
    for model_name in model_names:
        train_model(model_name)