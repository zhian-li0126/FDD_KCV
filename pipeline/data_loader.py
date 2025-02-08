import os
from pH_loading import load_prepare_diagnosis
from EC_loading import load_prepare_diagnosis

def load_data(model_name):
    """
    Loads the dataset based on model type.
    """
    if model_name.startswith("pH_"):
        x, y = load_prepare_diagnosis(split=False)
    elif model_name.startswith("EC_"):
        x, y = load_prepare_diagnosis(split=False)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return x, y