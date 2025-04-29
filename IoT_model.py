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
from utils import inject_faults
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
class IoT_model():
    
    def __init__(self, initial_data, thresh):
        self.init_pruning=0.9#0.9
        self.first_run=True
        self.model_name="autoencoder"
        self.trigger_threshold=thresh
        print("TensorFlow version:", tf.__version__)
        print("TFMOT version:", tfmot.__version__)
        self.initial_data=initial_data
        self.scaler=MinMaxScaler()

    def load_model(self):
        self.scaler = joblib.load(os.path.join("models", "scaler.pkl"))
        tflite_model_path = "models/"+self.model_name+".tflite"
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path, experimental_delegates=[])
        self.interpreter.allocate_tensors()
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.enc_in_shape=self.input_details[0]["shape_signature"]
        self.output_details = self.interpreter.get_output_details()
        self.dec_out_shape=self.output_details[0]["shape_signature"]

        if self.first_run:
            tensor_details = self.interpreter.get_tensor_details()
            total_memory = 0
            for tensor in tensor_details:
                shape = tensor['shape']
                dtype = tensor['dtype']
                size = np.prod(shape) * np.dtype(dtype).itemsize
                total_memory += size
            self.first_run=False
            print(f"Estimated total tensor memory: {total_memory / 1024:.2f} KB")


        
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

    def prepare_training_data(self, should_inject_faults=False, fit_scaler=False):
        def binary_label(y):
            return np.array([1 if label == 'BROKEN' else 0 for label in y])
        X=pd.read_csv(self.initial_data).drop(columns=["Unnamed: 0"], errors='ignore')
        y=X['machine_status']
        X=X.drop(columns=["timestamp", "machine_status"])
        if should_inject_faults:
            X, y = inject_faults(X,y, fault_fraction=0.4)

        y=binary_label(y)
        self.n_features = X.shape[1]  # number of sensors (~50)
        self.n_samples = len(X)
        if fit_scaler:
            print("FITTED SCALER")
            self.scaler.fit(X)
        X=self.scaler.transform(X)
        if fit_scaler:
            joblib.dump(self.scaler, os.path.join("models", "scaler.pkl"))
        return X, y


    def design_model_architecture(self):
        print("feats",self.n_features)
        inputs = tf.keras.Input(shape=(self.n_features,))
        encoded = tf.keras.layers.Dense(128, activation="relu")(inputs)
        encoded = tf.keras.layers.Dense(64, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(8, activation="relu")(encoded) 
        decoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(64, activation="relu")(decoded)
        decoded = tf.keras.layers.Dense(128, activation="relu")(decoded)

        decoded = tf.keras.layers.Dense(self.n_features, activation="linear")(decoded)

        # Create Functional Autoencoder Model
        autoencoder = tf.keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def make_model_quantization_aware(self, model):
        #quantized_model = tfmot.quantization.keras.quantize_model(model)
        #quantized_model.compile(optimizer='adam',loss='mse')
        #return quantized_model
        return model
    
    def make_representative_data(self, X):
        index = np.random.choice(X.shape[0], 1000, replace=False)
        x_random = X.iloc[index]
        return x_random

    def quantize_model(self, data,model, path, quantize=None):
        
        x_random=self.make_representative_data(pd.DataFrame(data))
        qm.quantize_8_bit(model,x_random, path, quantize)
        

    def check_sample(self, data):
        important=False
        w, h=np.shape(data)
        if w>1 and h>1:
            mse_val = max(mean_squared_error(self.scale_data(data).T, self.inference_on_model(data)))
        else:
            mse_val = mean_squared_error(self.scale_data(data).T, self.inference_on_model(data))
        if mse_val>self.trigger_threshold:
            important=True
        return important, mse_val

    def train_initial_model(self):
        batch_size=256
        epochs=20
        X, y = self.prepare_training_data(fit_scaler=True)
        total_steps=int(len(y)/batch_size*epochs)
        autoencoder = self.design_model_architecture()
        model = self.make_model_quantization_aware(autoencoder)
        ############################
        #pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,final_sparsity=self.init_pruning,begin_step=0,end_step=total_steps)}
        #model= tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        #model.compile(optimizer="adam", loss='mse')
        #callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        ##################################
        history = model.fit(
                X,X,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                #callbacks=callbacks
                )
        model = tfmot.sparsity.keras.strip_pruning(model)
        model.save(os.path.join("models", self.model_name+".h5"))
        self.quantize_model(X,model, os.path.join("models", self.model_name))
        

    #def combine_new_with_random_old(self, X, y, new, new_labels=None):
    #    old_data = X.sample(n=len(new), replace=False, random_state=42)
    #    new_data=pd.DataFrame(new)
    #    new_data.columns=old_data.columns
    #    data= pd.concat([new_data,old_data], axis=0)
    #    return data
    
    #def combine_faulty_with_random_old(self, new, new_labels=None):
    #    X_f=pd.read_csv("test_files/faulty_data.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    #    y_f=X_f['machine_status']
    #    X_f=X_f.drop(columns=["timestamp", "machine_status"])
    #    old_data = X_f.sample(n=len(new), replace=False, random_state=42)
    #    new_data=pd.DataFrame(new)
    #    new_data.columns=old_data.columns
    #    data= pd.concat([new_data,old_data])
    #    return data
    
    def combine_new_with_random_old(self, X, y, new, new_labels=None, num=100):
        # Sample old data and get corresponding labels
        old_data = X.sample(n=num, replace=False, random_state=42)
        old_indices = old_data.index
        if isinstance(y, pd.Series):
            old_labels = y.loc[old_indices]
        else:
            old_labels = y[old_indices]

        # Prepare new data
        new_data = pd.DataFrame(new)
        new_data.columns = old_data.columns

        # Concatenate features
        data = pd.concat([new_data, old_data], axis=0, ignore_index=True)

        # If labels provided, concatenate with old labels
        if new_labels is not None:
            combined_labels = pd.Series(list(new_labels) + list(old_labels), name=y.name if hasattr(y, "name") else "label")
            return data, combined_labels

        return data

    def combine_faulty_with_random_old(self, new, new_labels=None):
        # Load and prepare faulty data
        X_f = pd.read_csv("test_files/faulty_data.csv").drop(columns=["Unnamed: 0"], errors='ignore')
        y_f = X_f['machine_status']
        X_f = X_f.drop(columns=["timestamp", "machine_status"])
        y_f=binary_label(y_f)
        #set all labels to BROKEN to ensure more training data
        y_f[:] = 1
        # Sample old data and get corresponding labels
        if len(new) > len(X_f):
            n_samples = len(new)
            replace = True
        else:
            n_samples = len(new)
            replace = False
        old_data = X_f.sample(n=n_samples, replace=replace, random_state=42)

        if isinstance(y_f, pd.Series):
            old_labels = y_f.loc[old_data.index]
        else:
            old_labels = y_f[old_data.index]
        #old_labels = y_f.loc[old_data.index]

        # Prepare new data
        new_data = pd.DataFrame(new)
        new_data.columns = old_data.columns

        # Concatenate features
        data = pd.concat([new_data, old_data], ignore_index=True)

        # If labels provided, concatenate with old labels
        if new_labels is not None:
            combined_labels = pd.Series(list(new_labels) + list(old_labels), name="machine_status")
            return data, combined_labels

        return data

    def train_model(self, data, invert_loss=False, pruning_level=0):
        data = data.drop(data.columns[-1], axis=1)
        X, y = self.prepare_training_data()
        X=pd.DataFrame(X)
        data=np.array(data)
        new_data=self.scale_data(np.array(data))
        def mse_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            return -0.3*mse if invert_loss else mse  # Negate the loss to maximize
        
        with tfmot.quantization.keras.quantize_scope(), tf.keras.utils.custom_object_scope({'mse_loss': mse_loss}):
            model = tf.keras.models.load_model(os.path.join("models", self.model_name+".h5"))
        num_epochs = max(5, min(100, int(2000 / len(data))))

        if invert_loss==False:
            num_epochs=int(num_epochs)
        else:
            num_epochs=int(num_epochs/2)
        #for _ in range(num_epochs):

        ####
        if invert_loss==False:
            data=self.combine_new_with_random_old(X,y, new_data)
        elif os.path.getsize("test_files/faulty_data.csv") > 0:
            data=self.combine_faulty_with_random_old(new_data)
        else:
            data=new_data
        #####
        batch_size=128

        model.compile(optimizer="adam", loss=mse_loss)
        history =model.fit(data, data, epochs=num_epochs, batch_size=batch_size)
        return model, X

    #def prune_model(self, model,data, invert_loss=False, pruning_level=0):
        def mse_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            return -0.3*mse if invert_loss else mse  # Negate the loss to maximize
        batch_size=128
        num_epochs=10
        data = data.drop(data.columns[-1], axis=1)
        data=self.scale_data(np.array(data))
        #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-10)  # Almost no learning
        print(pruning_level)
        total_steps=int(len(data)/batch_size*num_epochs)
        #pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=max(pruning_level-0.1,0),final_sparsity=pruning_level,begin_step=0,end_step=10)}
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.9,    begin_step=0,    frequency=100)}
        # Wrap the pretrained model.
        model= tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model.compile(optimizer="adam", loss=mse_loss)
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        history = model.fit(data, data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks)
        print("Global step:", tf.keras.backend.get_value(tfmot.sparsity.keras.pruning_scheduler._global_step))
        for layer in model.layers:
            if hasattr(layer, 'pruning_vars'):
                pruning_vars = layer.pruning_vars
                for var in pruning_vars:
                    pass
                    #print(var)


        model = tfmot.sparsity.keras.strip_pruning(model)
        return model
    
    def manual_prune_weights(self, model, sparsity=0.9):
        """
        Manually zero out the lowest X% magnitude weights in each trainable kernel.
        :param model: a Keras model
        :param sparsity: float between 0 and 1, e.g., 0.9 means prune 90% of lowest weights
        :return: pruned model
        """
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                kernel = weights[0]
                # Flatten to compute threshold across all weights
                flat_kernel = np.abs(kernel).flatten()
                threshold = np.percentile(flat_kernel, sparsity * 100)

                pruned_kernel = np.where(np.abs(kernel) < threshold, 0, kernel)
                weights[0] = pruned_kernel
                layer.set_weights(weights)

                # Optional: print stats
                actual_sparsity = np.mean(pruned_kernel == 0)
                #print(f"{layer.name}: applied sparsity = {actual_sparsity:.2f}")
        return model

    def improve_model(self, data, invert_loss=False, pdr=0, throughput=None):
        #if invert_loss==True:
        #    return None
        #    return None
        #pruning_level=pdr
        quantize=False
        throughput=None
        if throughput:
            pruning_level=min(max(-0.84*(throughput/8 - 131.3)/100,0),0.95)
            if pruning_level>0.5:
                quantize=True
                pruning_level=min(max(-3.56*(throughput/8 - 42.17)/100,0),0.95)
            print("THROUGHPUT: ", throughput, "PRUNING: ", pruning_level, "Quantize, ", quantize)
        else:
            pruning_level=None
        #pruning_level=50
        model, X=self.train_model(data, invert_loss, pruning_level)
        if pruning_level:
            pruned_model = self.manual_prune_weights(model, pruning_level)
            print("Pruned model")
        model.save(os.path.join("models", self.model_name+".h5"))
        if pruning_level:
            self.quantize_model(X,pruned_model, os.path.join("models", self.model_name), quantize=quantize)
        else:
            self.quantize_model(X,model, os.path.join("models", self.model_name), quantize=quantize)


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