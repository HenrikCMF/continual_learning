import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from bin.utils import binary_label, get_string_config
import bin._quantize_model as qm
import joblib
from sklearn.metrics import mean_squared_error
from bin.utils import inject_faults
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, message=".*HDF5.*")
class IoT_model():
    def __init__(self, initial_data, thresh):
        """
        Initializes a fault detection model object

        Parameters:
        ----------
        initial_data : some data to train the initial model on          
        thresh : test parameter to set different trigger thresholds
        --------
        """
        self.init_pruning=0.9#0.9
        self.first_run=True
        self.model_name="autoencoder"
        self.trigger_threshold=thresh
        print("TensorFlow version:", tf.__version__)
        #print("TFMOT version:", tfmot.__version__)
        self.initial_data=initial_data
        self.scaler=MinMaxScaler()
        self.prepare_training_data(should_inject_faults=False, fit_scaler=True)

    def load_model(self):
        """
        Loads whichever autoencoder is currently at the hardcoded modelpath
        """

        config = get_string_config()
        self.scaler = joblib.load(os.path.join(config['file_paths']['models_dir'], config['file_paths']['scaler_file']))
        tflite_model_path = os.path.join(config['file_paths']['models_dir'], self.model_name + config['file_extensions']['tflite_extension'])

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

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        feature_cols=None,
        label_col: str = "machine_status",
        positive_label: str = "BROKEN",
        return_per_sample: bool = False,
        verbose_every: int = 0,
    ):
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]

        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)  # (N, 50)
        y = df[label_col].astype(str).to_numpy()
        n = X.shape[0]

        mse_vals = np.empty(n, dtype=np.float32)
        y_pred_pos = np.zeros(n, dtype=bool)
        y_true_pos = (y == positive_label)

        tp = fp = fn = tn = 0

        for i in range(n):
            # IMPORTANT: make it 2D so sklearn scaler (and your inference) won't crash
            x2d = X[i].reshape(1, -1)  # (1, 50)

            # TFLite inference (expects "one sample", but as 2D is fine: (1,50))
            recon = np.asarray(self.inference_on_model(x2d), dtype=np.float32).reshape(-1)  # (50,)

            # inference_on_model scales internally, so recon is in SCALED space
            x_scaled = np.asarray(self.scale_data(x2d), dtype=np.float32).reshape(-1)       # (50,)

            diff = x_scaled - recon
            mse_val = float(np.mean(diff * diff))

            mse_vals[i] = mse_val
            pred_pos = mse_val > self.trigger_threshold
            y_pred_pos[i] = pred_pos

            true_pos = y_true_pos[i]

            if pred_pos and true_pos:
                tp += 1
            elif pred_pos and not true_pos:
                fp += 1
            elif (not pred_pos) and true_pos:
                fn += 1
            else:
                tn += 1

            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"Processed {i+1}/{n} samples...")

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        summary = {
            "threshold": float(self.trigger_threshold),
            "positive_label": positive_label,
            "n_samples": int(n),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }

        if not return_per_sample:
            return summary

        per_sample = df[[label_col]].copy()
        per_sample["mse"] = mse_vals
        per_sample["pred_is_broken"] = y_pred_pos
        per_sample["true_is_broken"] = y_true_pos

        outcome = np.full(n, "TN", dtype=object)
        outcome[y_pred_pos & y_true_pos] = "TP"
        outcome[y_pred_pos & ~y_true_pos] = "FP"
        outcome[~y_pred_pos & y_true_pos] = "FN"
        per_sample["outcome"] = outcome

        return summary, per_sample


    def inference_on_model(self, data):
        """
        Uses the loaded model to inference on the given data

        Parameters:
        ----------
        data: a single sample     
        --------
        Returns:
        model output
        """
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
        """
        Function for the server to do data preparation on the initial training data

        Parameters:
        ----------
        should_inject_faults : When training a classifier, set to true to convert random points to faults.      
        fit_scaler : choose whether a scaler should be fitted on the training data or not.
        --------
        Returns:
        Training data X
        Training labels y
        """
        config = get_string_config()
        def binary_label(y):
            return np.array([1 if label == config['data_columns']['fault_label'] else 0 for label in y])
        X=pd.read_csv(self.initial_data).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        y=X[config['data_columns']['dataset_label']]
        X=X.drop(columns=[config['data_columns']['timestamp_column'], config['data_columns']['dataset_label']])
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
            joblib.dump(self.scaler, os.path.join(config['file_paths']['models_dir'], config['file_paths']['scaler_file']))
        return X, y


    def design_model_architecture(self):
        """
        Defines the architecture of the model

        Returns:
        Tensorflow model
        """
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
        qm.convert_to_tflite(model,x_random, path, quantize)
        

    def check_sample(self, data):
        """
        Finds the mean squared error between input and model output

        Parameters:
        ----------
        data : data sample  
        --------
        Returns:
        important = boolean, whether the mse is above trigger threshold or not
        mse_val = actual mse value.
        """
        important=False
        w, h=np.shape(data)
        if w>1 and h>1:
            mse_val = max(mean_squared_error(self.scale_data(data).T, self.inference_on_model(data)))
        else:
            mse_val = mean_squared_error(self.scale_data(data).T, self.inference_on_model(data))
        #print(mse_val)
        if mse_val>self.trigger_threshold:
            important=True
        return important, mse_val

    def train_initial_model(self):
        """
        Trains the initially defined model, with the initial dataset
        """
        batch_size=256
        epochs=20
        X, y = self.prepare_training_data(fit_scaler=True)
        total_steps=int(len(y)/batch_size*epochs)
        autoencoder = self.design_model_architecture()
        model = self.make_model_quantization_aware(autoencoder)
        history = model.fit(
                X,X,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
                )
        config = get_string_config()
        model.save(os.path.join(config['file_paths']['models_dir'], self.model_name + config['file_extensions']['h5_extension']))
        self.quantize_model(X,model, os.path.join(config['file_paths']['models_dir'], self.model_name))

    
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
        config = get_string_config()
        # Load and prepare faulty data
        X_f = pd.read_csv(os.path.join(config['file_paths']['test_files_dir'], config['file_paths']['faulty_data_file'])).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        X_f = X_f.loc[:, ~X_f.columns.str.startswith('Unnamed')]
        y_f = X_f[config['data_columns']['dataset_label']]
        X_f = X_f.drop(columns=[config['data_columns']['timestamp_column'], config['data_columns']['dataset_label']])
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
    #@tf.function(jit_compile=True)
    def train_model(self, data, invert_loss=False, input=-0.1):
        """
        Function to improve the most recent iteration of the model

        Parameters:
        ----------
        data : package of most recently received samples     
        invert_loss: test parameter
        --------
        Returns:
        model: improved tensorflow model
        X : data used to improve the model with
        """
        #data = data.drop(data.columns[-1], axis=1)
        X, y = self.prepare_training_data()
        X=pd.DataFrame(X)
        data=np.array(data)
        new_data=self.scale_data(np.array(data))
        def mse_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            return 0*mse if invert_loss else mse

        config = get_string_config()
        with tfmot.quantization.keras.quantize_scope(), tf.keras.utils.custom_object_scope({'mse_loss': mse_loss}):
            model = tf.keras.models.load_model(os.path.join(config['file_paths']['models_dir'], self.model_name + config['file_extensions']['h5_extension']))
        num_epochs = max(5, min(100, int(2000 / len(data))))

        if invert_loss==False:
            num_epochs=0#int(num_epochs)
        else:
            num_epochs=int(num_epochs/2)
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

    
    def manual_prune_weights(self, model, sparsity=0.9):
        """
        Manually prune the lowest X% of the weights by setting them to 0

        Parameters:
        ----------
        model : tenorflow model to be pruned     
        sparsity : X
        --------
        Returns:
        model: pruned model
        """
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                kernel = weights[0]
                flat_kernel = np.abs(kernel).flatten()
                threshold = np.percentile(flat_kernel, sparsity * 100)
                pruned_kernel = np.where(np.abs(kernel) < threshold, 0, kernel)
                weights[0] = pruned_kernel
                layer.set_weights(weights)
                actual_sparsity = np.mean(pruned_kernel == 0)
        return model

    def improve_model(self, data, invert_loss=False, throughput=None, t_DL=1):
            quantize=False
            config = get_string_config()
            if throughput:
                #pruning_level=min(max(-0.84*(throughput/8 - 140)/100,0),0.95)
                pruning_level=min(max(-0.8*(t_DL*throughput/8 - 125)/100,0),0.95)
                if pruning_level>0.4:
                    quantize=True
                    pruning_level=min(max(-3.57*(t_DL*throughput/8 - 44.4)/100,0),0.5)
                    if config['ablation_settings']['Link_adaptation_parts']['Quantization_enabled']==False:
                        quantize=False
                print("THROUGHPUT: ", throughput, "PRUNING: ", pruning_level, "Quantize, ", quantize)
            else:
                pruning_level=None
            #pruning_level=None
            model, X=self.train_model(data, invert_loss, input=input)
            model.save(os.path.join(config['file_paths']['models_dir'], self.model_name + config['file_extensions']['h5_extension']))
            if pruning_level and config['ablation_settings']['Link_adaptation_parts']['Pruning_enabled']:
                pruned_model = self.manual_prune_weights(model, pruning_level)
            
            
            if pruning_level:
                self.quantize_model(X,pruned_model, os.path.join(config['file_paths']['models_dir'], self.model_name), quantize=quantize)
            else:
                self.quantize_model(X,model, os.path.join(config['file_paths']['models_dir'], self.model_name), quantize=quantize)
            if quantize:
                return 8
            else:
                return 32

"""""
    def EECL_comp(self, throughput, model, X):
        quantize=False
        
        if throughput:
            pruning_level=min(max(-0.84*(throughput/8 - 140)/100,0),0.95)
            if pruning_level>0.4:
                quantize=True
                pruning_level=min(max(-4*(throughput/8 - 48)/100,0),0.8)
            pruned_model = self.manual_prune_weights(model, pruning_level)
            self.quantize_model(X,pruned_model, os.path.join("models", self.model_name), quantize=quantize)
            print("THROUGHPUT: ", throughput, "PRUNING: ", pruning_level, "Quantize, ", quantize)
        else:
            pruning_level=None
            self.quantize_model(X,model, os.path.join("models", self.model_name), quantize=quantize)
        if quantize:
            return 8
        else:
            return 32
        

    def improve_model(self, data, invert_loss=False, pdr=0, throughput=None):

        model, X=self.train_model(data, invert_loss)
        model.save(os.path.join("models", self.model_name+".h5"))

        quantization=self.EECL_comp(throughput, model, X)
        return quantization
"""""
