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
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
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
        X=pd.read_csv(self.initial_data).drop(columns=["Unnamed: 0"], errors='ignore')
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
        #encoded = tf.keras.layers.Dense(32, activation="relu")(inputs)
        #encoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
        #encoded = tf.keras.layers.Dense(8, activation="relu")(encoded)  # Bottleneck
        #decoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
        #decoded = tf.keras.layers.Dense(32, activation="relu")(decoded)

        encoded = tf.keras.layers.Dense(24, activation="relu")(inputs)
        encoded = tf.keras.layers.Dense(12, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(6, activation="relu")(encoded)  # Bottleneck
        decoded = tf.keras.layers.Dense(12, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(24, activation="relu")(decoded)

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
        x_random=self.make_representative_data(pd.DataFrame(data))
        qm.quantize_8_bit(model,x_random, path)

    def calc_mse(self, data):
        mse_val = mean_squared_error(self.scale_data(data).T, self.inference_on_model(data))
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

    def improve_model(self, data, invert_loss=False):
        X=pd.read_csv(self.initial_data).drop(columns=["Unnamed: 0"], errors='ignore')
        y=X['machine_status']
        X=X.drop(columns=["timestamp", "machine_status"])
        self.n_features = X.shape[1]  # number of sensors (~50)
        self.n_samples = len(X)
        #self.scaler.fit(X)
        X=self.scaler.transform(X)
        X=pd.DataFrame(X)
        data=np.array(data)
        new_data=self.scale_data(np.array(data))
        
        if invert_loss==False:
            old_data = X.sample(n=len(new_data), replace=False, random_state=42)
            new_data=pd.DataFrame(new_data)
            new_data.columns=old_data.columns
            data= pd.concat([new_data,old_data], axis=0)
        elif os.path.getsize("test_files/faulty_data.csv") > 0:
            X_f=pd.read_csv("test_files/faulty_data.csv").drop(columns=["Unnamed: 0"], errors='ignore')
            y_f=X_f['machine_status']
            X_f=X_f.drop(columns=["timestamp", "machine_status"])
            old_data = X_f.sample(n=len(new_data), replace=False, random_state=42)
            new_data=pd.DataFrame(new_data)
            new_data.columns=old_data.columns
            data= pd.concat([new_data,old_data])
        else:
            data=new_data

        def mse_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            return -mse if invert_loss else mse  # Negate the loss to maximize
        
        with tfmot.quantization.keras.quantize_scope(), tf.keras.utils.custom_object_scope({'mse_loss': mse_loss}):
            model = tf.keras.models.load_model(os.path.join("models", "autoencoder.h5"))

        model.compile(optimizer="adam", loss=mse_loss)
        num_epochs = max(5, min(100, int(4000 / len(data))))  # Scale epochs #8000
        if invert_loss==False:
            num_epochs=int(num_epochs/1)
        history =model.fit(data, data, epochs=num_epochs, batch_size=128)
        print("Final loss after training:", history.history['loss'][-1])
        model.save(os.path.join("models", "autoencoder.h5"))
        self.quantize_model(X,model, os.path.join("models", "autoencoder"))


#make_model=IoT_model("datasets/initial_data.csv")
#make_model.train_initial_model()
#make_model.load_model()
#df=pd.read_csv('datasets/initial_data.csv').drop(columns=["timestamp", "machine_status"])
#print(make_model.inference_on_batch(df))

#import AVRO
#data,timestamps, type, metadata = AVRO.load_AVRO_file("received/r_2018-04-20-02-30-00.avro")
#data=data.drop(data.columns[-1], axis=1)
#init_data=os.path.join('test_files','initial_data.csv')
#ml_model=IoT_model(init_data)
#data=np.array(data).reshape(50,-1)
#scaler = joblib.load(os.path.join("models", "scaler.pkl"))
#res=ml_model.scale_data(data)#.improve_model(data.drop(data.columns[-1], axis=1))
#print(pd.DataFrame(data))
#res=scaler.transform(data)

#print(res)