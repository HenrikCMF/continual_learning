import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from utils import binary_label
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
    def __init__(self, initial_data, thresh):
        super().__init__(initial_data, thresh)
        self.model_name="MLP_binary"
        self.trigger_threshold=0.16
    
    def check_sample(self, data):
        important=False
        score = self.inference_on_model(data)[0]
        if score>self.trigger_threshold:
            important=True
        return important, score

    def design_model_architecture(self):
        print("Custom architecture with", self.n_features, "features")
        inputs = tf.keras.Input(shape=(self.n_features,))
        x = tf.keras.layers.Dense(128, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        out=tf.keras.layers.Dense(1, activation='sigmoid')(x)

        autoencoder = tf.keras.Model(inputs, out)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    def train_initial_model(self):
        X, y = self.prepare_training_data(should_inject_faults=True, fit_scaler=True)
        model = self.design_model_architecture()
        history = model.fit(
            X, y,
            epochs=5,
            batch_size=128,
            verbose=1
        )
        model.save(os.path.join("models", self.model_name+".h5"))
        self.quantize_model(X, model, os.path.join("models", self.model_name))

    def train_model(self, data, invert_loss=False, pruning_level=0):
        data_labels= data.iloc[:, -1]
        data = data.drop(data.columns[-1], axis=1)
        if os.path.getsize("test_files/faulty_data.csv") > 0:
            X, y = self.prepare_training_data()
        else:
            X, y = self.prepare_training_data(should_inject_faults=True)
        X=pd.DataFrame(X)
        data=np.array(data)
        new_data=self.scale_data(np.array(data))
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(os.path.join("models", self.model_name+".h5"))
        num_epochs = max(5, min(1000, int(1000 / len(data))))
        if invert_loss:
            num_epochs*=2
        data_labels=binary_label(data_labels)
        if os.path.getsize("test_files/faulty_data.csv") > 0:
            new_data, data_labels=self.combine_faulty_with_random_old(new_data, data_labels)
        data, data_labels=self.combine_new_with_random_old(X,y, new_data, data_labels, num=int(len(new_data)/30))
        values1, counts1 = np.unique(data_labels, return_counts=True)
        print(dict(zip(values1, counts1)))
        print("about to train with input data of dim: ", np.shape(data), " with label number: ", np.shape(data_labels))
        
        model.compile(optimizer="adam", loss='binary_crossentropy')
        history =model.fit(data, data_labels, epochs=num_epochs, batch_size=128)
        return model, X