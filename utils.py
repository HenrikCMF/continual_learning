import socket
import time
import functools
import threading
import numpy as np
import pandas as pd
import json
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
def retry_transmission_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self_or_cls = args[0] if args else None
        is_method = hasattr(self_or_cls, '__class__')
        i=0
        while i<10:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.settimeout(None)
                    if is_method:
                        result = func(self_or_cls,client_socket,*args[1:], **kwargs)
                    else:
                        result = func(client_socket,*args, **kwargs)
                    break
            except Exception as e:
                print(e)
                time.sleep(0.1)
            i+=1
    return wrapper

def threaded(func):
    """Decorator to run a function in a new thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread  # Return the thread in case the caller wants to manage it
    return wrapper

def timed(func):
    def wrapper(*args, **kwargs):
        start=time.time()
        func(*args, **kwargs)
        stop=time.time()
        return stop-start
    return wrapper

def make_initial_data(path, out):
    df=pd.read_csv(path)
    sensors_to_drop = ['Unnamed: 0','sensor_15', 'sensor_50']
    df = df.drop(columns=sensors_to_drop)

    sensor_cols = df.columns[df.isnull().any()].tolist()
    df[sensor_cols] = df[sensor_cols].interpolate(method='linear')

    # If any remaining NaNs, use forward/backward fill
    df[sensor_cols] = df[sensor_cols].fillna(method='ffill')
    df[sensor_cols] = df[sensor_cols].fillna(method='bfill')
    y=df["machine_status"]
    X=df.drop(columns=['machine_status'])
    first_broken_idx = y[y == "BROKEN"].index[0]
    df.iloc[:first_broken_idx-200].to_csv(os.path.join(out,"initial_data.csv"),index=False)

def make_sensor_data(path):
    df=pd.read_csv(path)
    sensors_to_drop = ['Unnamed: 0','sensor_15', 'sensor_50']
    df = df.drop(columns=sensors_to_drop)

    sensor_cols = df.columns[df.isnull().any()].tolist()
    df[sensor_cols] = df[sensor_cols].interpolate(method='linear')

    # If any remaining NaNs, use forward/backward fill
    df[sensor_cols] = df[sensor_cols].fillna(method='ffill')
    df[sensor_cols] = df[sensor_cols].fillna(method='bfill')
    y=df["machine_status"]
    X=df.drop(columns=['machine_status'])
    first_broken_idx = y[y == "BROKEN"].index[0]
    df.iloc[first_broken_idx-200:].to_csv("data_to_be_measured.csv",index=False)

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.feature_range = feature_range

    def fit(self, X):
        """Compute the min and max values for scaling"""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

    def transform(self, X):
        """Scale the input X based on the computed min-max values"""
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled

    def fit_transform(self, X):
        """Fit and transform the data"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling transformation"""
        X = (X_scaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        X = X * (self.max_ - self.min_) + self.min_
        return X
    
def generate_avro_schema(datalength, filename):
    schema = {
        "type": "record",
        "name": "SensorData",
        "fields": [
            {"name": "timestamp", "type": "string"}
        ]
    }
    
    # Add 50 sensor data fields with numeric names
    for i in range(datalength-1):
        schema["fields"].append({"name": str(i), "type": "float"})
    schema["fields"].append({"name": str(i+1), "type": "string"})
    # Write schema to file
    with open(filename, "w") as f:
        json.dump(schema, f, indent=4)

def make_dataset(fault_index, num):
    df=pd.read_csv("datasets/sensor.csv")
    #sensors_to_drop = ['Unnamed: 0', 'timestamp','sensor_15', 'sensor_50']
    sensors_to_drop = ['Unnamed: 0','sensor_15', 'sensor_50']
    df = df.drop(columns=sensors_to_drop)
    sensor_cols = df.columns[df.isnull().any()].tolist()
    df[sensor_cols] = df[sensor_cols].interpolate(method='linear')
    # If any remaining NaNs, use forward/backward fill
    df[sensor_cols] = df[sensor_cols].fillna(method='ffill')
    df[sensor_cols] = df[sensor_cols].fillna(method='bfill')
    y=df["machine_status"]
    broken_idx = y[y == "BROKEN"].index[fault_index]
    filename="test_files/initial_data"+str(num)+".csv"
    df.iloc[broken_idx-100:].to_csv(filename,index=False)
    return filename, broken_idx-100

def remove_all_avro_files(path):
    directory = Path(path)
    for file in directory.glob("*.avro"):
        file.unlink()


def make_end_plot(mse, offset):
    file_path = "datasets/sensor.csv"
    df = pd.read_csv(file_path)

    # Ensure 'machine_status' column exists
    if "machine_status" not in df.columns:
        raise ValueError("Column 'machine_status' not found in dataset")

    # Find indices where machine_status is 'BROKEN'
    broken_indices = df.index[df["machine_status"] == "BROKEN"].tolist()
    last_index = df.index[-1]
    adjusted_broken_indices = [idx - offset for idx in broken_indices if idx >= offset]

    mse_buf = []
    for i in range(len(mse)):
        mse_buf.append(pd.read_csv(mse[i]).iloc[:,1])

    # Plot the mse_buf values
    plt.figure(figsize=(10, 5))
    #for i in mse_buf:
    
    plt.plot(mse_buf[0], label="MSE, Continual Learning", alpha=0.8)
    plt.plot(mse_buf[1], label="MSE, Continual Learning, 7 batches combined", alpha=0.8)
    plt.ylim(None, 10)
    # Plot vertical lines where 'machine_status' is 'BROKEN'
    for idx in adjusted_broken_indices:
        plt.axvline(x=idx, color='r', linestyle='--', alpha=0.3, label="Fault Instances" if idx == adjusted_broken_indices[0] else "")
    #plt.axvline(x=(last_index-offset), color='g', linestyle='--', alpha=0.7, label="Last Entry")


    # Labels and legend
    plt.xlabel("Index")
    plt.ylabel("MSE Values")
    plt.title("Model Reconstruction Error over dataset")
    plt.legend()
    plt.show()

def binary_label(y):
    return np.array([1 if label == 'BROKEN' else 0 for label in y])

def inject_faults(x, y, fault_fraction=0.1, decrease_fraction=0.3, decrease_value=0.2):
    # Create copies to avoid modifying original data
    x_fault = x.copy()
    y_fault = y.copy()

    # Determine shape and feature count based on type
    if isinstance(x_fault, pd.DataFrame):
        n_samples, n_features = x_fault.shape
        is_df = True
    else:
        n_samples, n_features = x_fault.shape
        is_df = False

    # Determine number of samples to mark as faulty
    n_faulty = int(n_samples * fault_fraction)

    # Randomly select indices for the samples that will become faulty
    faulty_indices = np.random.choice(n_samples, size=n_faulty, replace=False)

    for idx in faulty_indices:
        # Mark the sample as faulty in y
        if isinstance(y_fault, pd.Series):
            y_fault.iloc[idx] = "BROKEN"
        else:
            y_fault[idx] = "BROKEN"

        # Determine the number of features to decrease
        n_decrease = max(1, int(np.ceil(n_features * decrease_fraction)))

        # Randomly select feature indices
        feature_indices = np.random.choice(n_features, size=n_decrease, replace=False)

        # Apply the decrease
        if is_df:
            cols = x_fault.columns[feature_indices]
            x_fault.loc[idx, cols] *= (1 - decrease_value)
        else:
            x_fault[idx, feature_indices] *= (1 - decrease_value)

    return x_fault, y_fault