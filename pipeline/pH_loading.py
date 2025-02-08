import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def load_prepare_diagnosis(filepath = "pH_average.csv"):
    '''
    Generate training and testing dataset based on the targeted csv
    
    Parameters:
    filepath: str
    The path contains the csv file you want to convert to diagnosis dataset
    '''
    # Step1: load data sheet
    dataset = pd.read_csv(filepath)
    # Step2: define parameters
    # 7 classes of fault types (5 types, 2+2+1+1+1=7)
    labels = ['pH','pH_bias_up', 'pH_bias_low', 'pH_drift_up', 'pH_drift_low', 'pH_precision', 'pH_stuck', 'pH_spike']
    #labels = ['pH_bias_up', 'pH_bias_low', 'pH_drift_up', 'pH_drift_low', 'pH_prpHision', 'pH_stuck', 'pH_spike']
    # define pH properties
    repeat = ['pH_lower', 'pH_upper','pH(t-1)','pH(t-2)','pH(t-3)']
    # define fault status (0 = normal, 1 = faulty)
    test = ['test_normal','test_drift_up','test_drift_low','test_bias_up','test_bias_low'
                ,'test_precision','test_stuck','test_spike']
    serial = ['serial_normal','serial_drift_up','serial_drift_low','serial_bias_up', 'serial_bias_low'
                ,'serial_precision','serial_stuck','serial_spike']
    # convert data type of some columns to int
    convert_list = test + serial
    convert_dict = {i: 'int' for i in convert_list}
    dataset = dataset.astype(convert_dict)

    # preprocess the data sheet, concatenating, etc
    repeat_value = dataset[repeat].values
    classifier = []
    classifier = [repeat_value] * len(labels)
    classifier = np.concatenate(classifier)
    #print(classifier)
    label_value = []
    test_value = []
    serial_value =[]
    for i in range(len(labels)):
        label_value.append(dataset[labels[i]].values)
        test_value.append(dataset[test[i]].values)
        serial_value.append(dataset[serial[i]].values)
    label_value = np.concatenate(label_value)
    test_value = np.concatenate(test_value)
    serial_value = np.concatenate(serial_value)
    #print(label_value)
    X = np.insert(classifier, 5, label_value, axis=1)
    # create the class of five faults
    array = []
    for i in range(len(labels)):
        array.append(np.full((70373, 1), i))
    array = np.concatenate(array)
    #X = np.insert(X, 4, test_value, axis=1)
    Y = serial_value
    #
    Y = tf.keras.utils.to_categorical(Y)
    smote = SMOTE(sampling_strategy='all')
    X_res, y_res = smote.fit_resample(X,Y)


    '''# Step 3: standardize the data and randomly split the train and test spHtions
    from sklearn.preprocessing import StandardScaler
    PredictorScaler=StandardScaler()
    PredictorScalerFit=PredictorScaler.fit(X)

    # Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)'''

    # Split the data into training and testing set
    #X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=64)
    #train, test, val = random_split(Dataset, [0.6, 0.3, 0.1])
    return X_train, X_test, y_train, y_test


def load_prepare_detection(filepath = "pH_average.csv"):
    '''
    Generate training and testing dataset based on the targeted csv
    
    Parameters:
    filepath: str
    The path contains the csv file you want to convert to detection dataset
    '''
    # Step1: load data sheet
    dataset = pd.read_csv(filepath)
    # Step2: define parameters
    # 7 classes of fault types (5 types, 2+2+1+1+1=7)
    labels = ['pH','pH_bias_up', 'pH_bias_low', 'pH_drift_up', 'pH_drift_low', 'pH_precision', 'pH_stuck', 'pH_spike']
    #labels = ['pH_bias_up', 'pH_bias_low', 'pH_drift_up', 'pH_drift_low', 'pH_prpHision', 'pH_stuck', 'pH_spike']
    # define pH properties
    repeat = ['pH_lower', 'pH_upper','pH(t-1)','pH(t-2)','pH(t-3)']
    # define fault status (0 = normal, 1 = faulty)
    test = ['test_normal','test_drift_up','test_drift_low','test_bias_up','test_bias_low'
            ,'test_precision','test_stuck','test_spike']
    repeat_value = dataset[repeat].values
    classifier = []
    classifier = [repeat_value] * len(labels)
    classifier = np.concatenate(classifier)
    #print(classifier)
    label_value = []
    test_value = []
    for i in range(len(labels)):
        label_value.append(dataset[labels[i]].values)
        test_value.append(dataset[test[i]].values)
    label_value = np.concatenate(label_value)
    test_value = np.concatenate(test_value)
    #print(label_value)
    X = np.insert(classifier, 3, label_value, axis=1)
    Y = test_value

    # Step 3: standardize the data and randomly split the train and test sections
    smote = SMOTE(sampling_strategy='all')
    X_res, y_res = smote.fit_resample(X,Y)

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=64)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    x1, x2, y1, y2 = load_prepare_diagnosis()
    print(x1[-1])
