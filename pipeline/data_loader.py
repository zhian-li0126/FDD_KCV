# data_loader.py

import os

# If these exist, import your actual diagnosis/detection loading fns:
from pH_loading import load_prepare_diagnosis_pH, load_prepare_detection_pH
from EC_loading import load_prepare_diagnosis_EC, load_prepare_detection_EC


def load_data(domain, model_name):
    """
    Loads the dataset based on domain ('diagnosis' or 'detection')
    and the model name (which starts with 'pH_' or 'EC_').

    Returns (x, y) as NumPy arrays.
    """
    if domain not in ("diagnosis", "detection"):
        raise ValueError(f"domain must be 'diagnosis' or 'detection', got: {domain}")

    # Decide pH vs EC by checking model_name prefix
    if model_name.startswith("pH_"):
        # pH models
        if domain == "diagnosis":
            x, y = load_prepare_diagnosis_pH(split=False)
        else:  # domain == "detection"
            x, y = load_prepare_detection_pH(split=False)
    elif model_name.startswith("EC_"):
        # EC models
        if domain == "diagnosis":
            x, y = load_prepare_diagnosis_EC(split=False)
        else:  # domain == "detection"
            x, y = load_prepare_detection_EC(split=False)
    else:
        raise ValueError(f"Unknown model type name: {model_name}")

    return x, y
