import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot
#from utils import MinMaxScaler
import _quantize_model as qm
import os
import joblib
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
from IoT_model import IoT_model
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

class mlp_classifier(IoT_model):
    def __init__(self, initial_data):
        self.model_name="MLP_binary"
        # Inherit everything else
        super().__init__(initial_data)
    
    def design_model_architecture(self):
        print("Custom architecture with", self.n_features, "features")
        inputs = tf.keras.Input(shape=(self.n_features,))
        
        # Custom architecture (e.g., deeper encoder/decoder)
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)  # Bottleneck
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        out=tf.keras.layers.Dense(1, activation='sigmoid')

        autoencoder = tf.keras.Model(inputs, out)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    def train_initial_model(self):
        # Optionally override training behavior here too
        X, y = self.prepare_training_data()
        model = self.design_model_architecture()
        history = model.fit(
            X, y,
            epochs=10,  # shorter training for testing
            batch_size=128,
            verbose=1
        )
        model.save(os.path.join("models", self.model_name+".h5"))
        self.quantize_model(X, model, os.path.join("models", self.model_name))