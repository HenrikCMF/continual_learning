import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric")
from TCP_code import TCP_COM
import json
import time
import queue
from network_control import network_control
import IoT_model
from alternative_iot_models import mlp_classifier
import AVRO
import os
from utils import make_initial_data, remove_all_avro_files
import numpy as np
import zipfile
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import subprocess
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

#Main File for the server Node


class Base_station(TCP_COM):
    #Initialize Server
    def __init__(self, REC_FILE_PATH, input, bQ, eQ):
        """
        Initializes a server listening and model training object..

        Parameters:
        ----------
        REC_FILE_PATH : string acting as path to folder where received files are to be stored.           
        input : test parameter.
        --------
        """
        self.total_data_sent=0
        self.throughputs=[]
        self.use_PDR=False
        self.NEW_START=True
        self.faulty_data=os.path.join('test_files','faulty_data.csv')
        #Fetch only the "Allowed" part of the training data, according to the rules
        if self.NEW_START:
            make_initial_data("datasets/sensor.csv", 'test_files')
            with open(self.faulty_data, 'w') as f:
                f.write("")
        
        self.init_data=os.path.join('test_files','initial_data.csv')
        self.init_data_columns=pd.read_csv(self.init_data).drop(columns=["Unnamed: 0"], errors='ignore').columns
        #Load the chosen model
        self.ml_model=IoT_model.IoT_model(self.init_data, 0.2) #Autoencoder
        if self.NEW_START:
            #Train the initial model
            self.ml_model.train_initial_model()
        self.device_type="bs"
        #Load all configs to set up as TCP server.
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.file_Q=queue.Queue()
        self.configs=configs
        super().__init__(bQ,eQ, REC_FILE_PATH, self.device_type, self.file_Q,throughput=input)
    
    def append_to_initial_data(self, data, timestamps, init_data_path):
        """
        Appends new data with corresponding timestamps to the initial data CSV file.

        Parameters:
        ----------
        data : must be a pd.Dataframe, containing recent sensor data. Must align column-wise with `self.init_data_columns`,
            excluding the timestamp.           
        timestamps : must be array-like corresponding to each row in `data`.
        init_data_path : str path to the CSV file where the combined data and timestamps will be appended.
        --------
        """
        timestamps=pd.DataFrame(timestamps)
        timestamps.columns=['timestamp']
        df2 = pd.concat([timestamps, data], axis=1).drop(columns=["Unnamed: 0"], errors='ignore')
        df2.columns=self.init_data_columns
        df2.to_csv(init_data_path, mode='a', header=False, index=False)

    def append_to_faulty_data(self, data, timestamps, init_data_path):
        """
        Appends new faulty data with corresponding timestamps to a growing database of faults.

        Parameters:
        ----------
        data : must be a pd.Dataframe, containing recent sensor data. Must align column-wise with previous faults,
            excluding the timestamp.           
        timestamps : must be array-like corresponding to each row in `data`.
        init_data_path : str path to the CSV file where the combined data and timestamps will be appended.
        --------
        """
        init_data_no_faults=pd.read_csv(self.init_data).drop(columns=["Unnamed: 0"], errors='ignore')
        if os.path.getsize(init_data_path) <= 0:
            init_data=pd.DataFrame()
        else:
            init_data=pd.read_csv(init_data_path, on_bad_lines='skip').drop(columns=["Unnamed: 0"], errors='ignore')
        timestamps=pd.DataFrame(timestamps)
        timestamps.columns=['timestamp']
        df2 = pd.concat([timestamps, data], axis=1).drop(columns=["Unnamed: 0"], errors='ignore')
        df2.columns=init_data_no_faults.columns
        df_combined = pd.concat([init_data, df2], ignore_index=True).drop(columns=["Unnamed: 0"], errors='ignore')
        df_combined.to_csv(init_data_path)

    def run(self, input, resultQ):
        """
        Runs the basic server routine of waiting for samples, using them to improve the model, then transmitting the improved model back.

        Parameters:
        ----------
        input: test parameter
        --------
        Returns:
        TP: Total number of received packages containing a fault
        FP: Total number of received packages not containing a fault
        Average throughput: Average of all measured throughputs.
        """
        Running=True
        TP=0
        FP=0
        start=time.time()
        #Wait for X clients:
        clients=0
        while clients<1:
            file, transmission_time = self.file_Q.get(timeout=None, block=True)
            clients+=1
        self.distribute_model("models/"+self.ml_model.model_name+".tflite")
        
        while Running:
            try:
                file, transmission_time = self.file_Q.get(timeout=3)
                if isinstance(file, str):
                    if file=="DONE":
                        Running=False
                    self.file_Q.task_done()
                    if "ACK" in file:
                        self.distribute_model("models/"+self.ml_model.model_name+".tflite")
                else:
                    data,timestamps, batch_num=file
                    timestamps=pd.DataFrame(timestamps)
                    data=pd.DataFrame(data)
                    batches = np.array_split(data, batch_num)
                    #print("batches: ",batches)
                    for i, batch in enumerate(batches):
                        invert_training=False
                        if batch.iloc[:, -1].eq("BROKEN").any():
                            print("INVERTED TRAINING")
                            invert_training=True
                            TP+=1
                        else:
                            FP+=1
                        #self.ml_model.improve_model(batch.drop(batch.columns[-1], axis=1), invert_training, pdr=self.PDR)
                        #if invert_training==False:
                        self.throughputs.append(self.throughput)
                        self.model_quantization=self.ml_model.improve_model(batch, invert_training, pdr=self.PDR, throughput=self.throughput)
                        if invert_training==False:
                            self.append_to_initial_data(data, timestamps, self.init_data)
                        else:
                            self.append_to_faulty_data(data, timestamps, self.faulty_data)
                    #if self.model_quantization<32:
                    #    prefix=""
                    #else:
                    #    prefix=""
                    print("distributing", self.ml_model.model_name)
                    self.distribute_model("models/"+self.ml_model.model_name+".tflite")
                    #self.rate_kbps-=10
                    #self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
            except queue.Empty:
                #print("waiting for data")
                pass
        resultQ.put({"TP":TP, "FP":FP, "avg_throughput":np.mean(self.throughputs)})
            #
        #self.send_file("307.jpg")

    def distribute_model(self, model):
        """
        Compresses a received model to a zip file and transmits it to all known IoT receivers.

        Parameters:
        ----------
        data : must be a pd.Dataframe, containing recent sensor data. Must align column-wise with previous faults,
            excluding the timestamp.           
        timestamps : must be array-like corresponding to each row in `data`.
        init_data_path : str path to the CSV file where the combined data and timestamps will be appended.
        --------
        """
        output_zip=model+'.zip'
        input_file=model
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(input_file, arcname=os.path.basename(input_file))
        self.total_data_sent+=os.path.getsize(output_zip)
        self.send_file("",output_zip)
            
            #self.send_file(ip, self.TAR_PORT_TCP,model)
            #self.send_file(ip, self.TAR_PORT_TCP,"models/autoencoder.h5")

#bs=Base_station("received", 1000)
#bs.run(1000)