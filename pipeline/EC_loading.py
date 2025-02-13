import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_prepare_diagnosis_EC(filepath= os.path.join(os.getcwd() + "/pipeline/utl/EC_average.csv"), split=True, scale_features=True, apply_smote=True):
    """
    Loads and preprocesses the EC dataset for fault diagnosis.

    Parameters:
    - filepath (str): Path to the CSV file.
    - split (bool): Whether to split the dataset into training and test sets.
    - scale_features (bool): Whether to normalize input features using MinMaxScaler.
    - apply_smote (bool): Whether to apply SMOTE for class balancing.

    Returns:
    - (X_train, X_test, y_train, y_test) if split=True
    - (X_res, y_res) if split=False
    """

    # Load dataset
    dataset = pd.read_csv(filepath)

    # Define feature and diagnosis columns
    feature_cols = ['EC_lower', 'EC_upper', 'EC(t-1)', 'EC(t-2)', 'EC(t-3)', 'EC']
    diagnosis_cols = ['EC', 'EC_bias_up', 'EC_bias_low', 'EC_drift_up', 'EC_drift_low', 
                        'EC_precision', 'EC_stuck', 'EC_spike']

    # Define the desired order for fault classes.
    fault_order = ["normal", "bias_up", "bias_low", "drift_up", "drift_low", "precision", "stuck", "spike"]

    # Create the fault classes using the corresponding "serial" columns in your CSV
    fault_classes = {
        "normal": dataset["serial_normal"],
        "bias_up": dataset["serial_bias_up"], "bias_low" : dataset["serial_bias_low"],
        "drift_up": dataset["serial_drift_up"], "drift_low" : dataset["serial_drift_low"],
        "precision": dataset["serial_precision"],
        "stuck": dataset["serial_stuck"],
        "spike": dataset["serial_spike"]
    }

    # Create a list of column groups: each group is feature columns + one diagnosis column.
    temp_cols = []
    for diag in diagnosis_cols:
        temp_col = feature_cols + [diag]
        temp_cols.append(temp_col)
    # Process each group: extract the columns and rename the diagnosis column to "diagnosis_value".
    data_blocks = []
    for diag, cols in zip(diagnosis_cols, temp_cols):
        block = dataset[cols].copy().values
        data_blocks.append(block)

    # If you want to separate the features and labels:
    X = np.concatenate(data_blocks)

    # Start with an empty row vector (1,0)
    y_list = np.zeros((1, 0), dtype=int)
        
    for idx, fault in enumerate(fault_order):
        # Convert the fault column (a Pandas Series) to a boolean mask list.
        col = np.array(fault_classes[fault].values.tolist()).reshape(1, -1)
        
        # Append this row vector to y along the columns (axis=1)
        y_list = np.append(y_list, col, axis=1)
        y  = y_list.reshape(-1, 1).flatten()


    # Scale input features
    if scale_features:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Apply SMOTE for class balancing
    if apply_smote:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X, y = smote.fit_resample(X, y)

    # Split the dataset
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def load_prepare_detection_EC(filepath= os.path.join(os.getcwd() + "/pipeline/utl/C_average.csv"), split=True, scale_features=True, apply_smote=True):
    """
    Loads and preprocesses the EC dataset for fault detection.

    Parameters:
    - filepath (str): Path to the CSV file.
    - split (bool): Whether to split the dataset into training and test sets.
    - scale_features (bool): Whether to normalize input features using MinMaxScaler.
    - apply_smote (bool): Whether to apply SMOTE for class balancing.

    Returns:
    - (X_train, X_test, y_train, y_test) if split=True
    - (X_res, y_res) if split=False
    """
    # Load dataset
    dataset = pd.read_csv(filepath)

    # Define feature and diagnosis columns
    feature_cols = ['EC_lower', 'EC_upper', 'EC(t-1)', 'EC(t-2)', 'EC(t-3)', 'EC']
    diagnosis_cols = ['EC', 'EC_bias_up', 'EC_bias_low', 'EC_drift_up', 'EC_drift_low', 
                        'EC_precision', 'EC_stuck', 'EC_spike']

    # Define the desired order for fault classes.
    fault_order = ["normal", "bias_up", "bias_low", "drift_up", "drift_low", "precision", "stuck", "spike"]

    # Create the fault classes using the corresponding "serial" columns in your CSV
    fault_classes = {
        "normal": dataset["test_normal"],
        "bias_up": dataset["test_bias_up"], "bias_low" : dataset["test_bias_low"],
        "drift_up": dataset["test_drift_up"], "drift_low" : dataset["test_drift_low"],
        "precision": dataset["test_precision"],
        "stuck": dataset["test_stuck"],
        "spike": dataset["test_spike"]
    }

    # Create a list of column groups: each group is feature columns + one diagnosis column.
    temp_cols = []
    for diag in diagnosis_cols:
        temp_col = feature_cols + [diag]
        temp_cols.append(temp_col)
    # Process each group: extract the columns and rename the diagnosis column to "diagnosis_value".
    data_blocks = []
    for diag, cols in zip(diagnosis_cols, temp_cols):
        block = dataset[cols].copy().values
        data_blocks.append(block)

    # If you want to separate the features and labels:
    X = np.concatenate(data_blocks)

    # Start with an empty row vector (1,0)
    y_list = np.zeros((1, 0), dtype=int)
        
    for idx, fault in enumerate(fault_order):
        # Convert the fault column (a Pandas Series) to a boolean mask list.
        col = np.array(fault_classes[fault].values.tolist()).reshape(1, -1)
        
        # Append this row vector to y along the columns (axis=1)
        y_list = np.append(y_list, col, axis=1)
        y  = y_list.reshape(-1, 1).flatten()


    # Scale input features
    if scale_features:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Apply SMOTE for class balancing
    if apply_smote:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X, y = smote.fit_resample(X, y)

    # Split the dataset
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        return X, y

if __name__ == "__main__":
    result = load_prepare_diagnosis_EC()  # Default is split=True
    print(len(result))

    if len(result) == 2:  # When split=False
        a, b = result
        print(a.shape, b.shape)
    else:  # When split=True
        a, b, c, d = result
        print(a.shape, b.shape, c.shape, d.shape)
