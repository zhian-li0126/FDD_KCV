import importlib
import os
import pandas as pd
from data_loader import load_data

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'


# List of model file names (without .py extension)
diagnosis_models = ["pH_LSTM", "pH_ANN", "pH_RF", "EC_LSTM", "EC_ANN", "EC_RF"]
detection_models = ["pH_LSTM", "pH_ANN", "pH_RF", "EC_LSTM", "EC_ANN", "EC_RF"]


# Training function
def train_model(domain: str, model_name: str):
    """
    Dynamically imports a model from models/{domain}/{model_name},
    loads the dataset, and starts training.
    """
    print(f"\n🚀 Training model: {model_name} in domain: {domain} ...")

    # Load data
    x, y = load_data(domain, model_name)

    # Dynamically import the correct subfolder + model
    #    e.g. "models.diagnosis.pH_LSTM" or "models.detection.EC_ANN"
    module_path = f"models.{domain}.{model_name}"
    model_module = importlib.import_module(module_path)

    # Instantiate the model
    model = model_module.Model()

    # Train model
    result = model.train_model(x, y)
    
    # Save model
    df = pd.DataFrame(result)
    output_dir = os.path.join("data", "result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.csv")

    # Write to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Training complete for: {model_name}\n model saved to {os.getcwd()}\{model_name}.csv\n")

    print(f"✅ Training complete for: {model_name}\n model saved to {os.getcwd()}\{model_name}.csv\n")

# Run all models sequentially
def main():
    # Example usage: train all diagnosis models, then all detection models
    for m in diagnosis_models:
        train_model("diagnosis", m)

    for m in detection_models:
        train_model("detection", m)

if __name__ == "__main__":
    main()
