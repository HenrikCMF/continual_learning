import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from utils import MinMaxScaler
import _quantize_model as qm
import os
import joblib
from sklearn.metrics import mean_squared_error
class IoT_model():
    
    def __init__(self, initial_data):
        print("TensorFlow version:", tf.__version__)
        print("TFMOT version:", tfmot.__version__)
        self.initial_data=initial_data
        self.scaler=MinMaxScaler()

    def load_model(self):
        self.scaler = joblib.load(os.path.join("models", "scaler.pkl"))
        tflite_model_path = "models/autoencoder.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.enc_in_shape=self.input_details[0]["shape_signature"]
        self.output_details = self.interpreter.get_output_details()
        self.dec_out_shape=self.output_details[0]["shape_signature"]
        
    def inference_on_model(self, data):
        data=np.array(self.scale_data(data))
        self.interpreter.set_tensor(self.input_details[0]['index'], np.reshape(data.astype(np.float32),(-1,self.enc_in_shape[1])))
        self.interpreter.invoke()
        output_data = np.reshape(self.interpreter.get_tensor(self.output_details[0]['index']),(self.dec_out_shape[1],-1))
        output_data=output_data.reshape(1,-1)
        return list(output_data[0])
    
    def inference_on_batch(self, data):
        data=np.array(data)
        results=[]
        print("data shape", np.shape(data))
        for i in range(np.shape(data)[1]):
            results.append(self.inference_on_model(i))
        return results
    
    def scale_data(self, data):
        return self.scaler.transform(data)
    def unscale_data(self, data):
        return self.scaler.inverse_transform(data)

    def prepare_training_data(self):
        X=pd.read_csv(self.initial_data)
        y=X['machine_status']
        X=X.drop(columns=["timestamp", "machine_status"])
        self.n_features = X.shape[1]  # number of sensors (~50)
        self.n_samples = len(X)
        self.scaler.fit(X)
        X=self.scaler.transform(X)
        joblib.dump(self.scaler, os.path.join("models", "scaler.pkl"))
        return X, y


    def design_model_architecture(self):
        print("feats",self.n_features)
        inputs = tf.keras.Input(shape=(self.n_features,))
        # Encoder
        encoded = tf.keras.layers.Dense(32, activation="relu")(inputs)
        encoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(8, activation="relu")(encoded)  # Bottleneck
        decoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(32, activation="relu")(decoded)
        decoded = tf.keras.layers.Dense(self.n_features, activation="linear")(decoded)

        # Create Functional Autoencoder Model
        autoencoder = tf.keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def make_model_quantization_aware(self, model):
        quantized_model = tfmot.quantization.keras.quantize_model(model)
        quantized_model.compile(optimizer='adam',loss='mse')
        return quantized_model
    
    def make_representative_data(self, X):
        index = np.random.choice(X.shape[0], 100, replace=False)
        x_random = X.iloc[index]
        return x_random

    def quantize_model(self, data,model, path):
        x_random=self.make_representative_data(data)
        qm.quantize_8_bit(model,x_random, path)

    def calc_mse(self, data):
        mse_val = mean_squared_error(self.scale_data(data), self.inference_on_model(data))
        return mse_val

    def train_initial_model(self):
        X, y = self.prepare_training_data()
        autoencoder = self.design_model_architecture()
        model = self.make_model_quantization_aware(autoencoder)
        history = model.fit(
                X,X,
                epochs=20,
                batch_size=256,
                verbose=1
                )
        model.save(os.path.join("models", "autoencoder.h5"))
        self.quantize_model(X,model, os.path.join("models", "autoencoder"))

    def improve_model(self, data):
        model = tf.keras.models.load_model(os.path.join("models", "autoencoder.h5"))
        model.compile(optimizer="adam", loss="mse")
        model.fit(data, data, epochs=1, batch_size=128)
        model.save(os.path.join("models", "autoencoder.h5"))
        X, y = self.prepare_training_data()
        self.quantize_model(X,model, os.path.join("models", "autoencoder"))


#make_model=IoT_model("datasets/initial_data.csv")
#make_model.train_initial_model()
#make_model.load_model()
#df=pd.read_csv('datasets/initial_data.csv').drop(columns=["timestamp", "machine_status"])
#print(make_model.inference_on_batch(df))

