import socket
import time
import functools
import threading
import numpy as np
import pandas as pd
def retry_transmission_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self_or_cls = args[0] if args else None
        is_method = hasattr(self_or_cls, '__class__')
        i=0
        while i<10:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
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

def make_initial_data(path):
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
    df.iloc[:first_broken_idx-200].to_csv("initial_data.csv",index=False)

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