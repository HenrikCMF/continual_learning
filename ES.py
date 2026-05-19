from bin.TCP_code import TCP_COM
import json
import time
import queue
from bin.network_control import network_control
import bin.AVRO as AVRO
import os
from bin.utils import make_initial_data, remove_all_avro_files, get_string_config
import numpy as np
import zipfile
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
import subprocess
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

def _label_matches(v, target):
    sv, st = str(v).strip(), str(target).strip()
    if sv == st:
        return True
    try:
        return float(sv) == float(st)
    except (ValueError, TypeError):
        return False

#Main File for the server Node


class ES_station(TCP_COM):
    #Initialize Server
    def __init__(self, REC_FILE_PATH, bandwidth=1000, energy_budget=60):
        """
        Initializes a server listening and model training object..

        Parameters:
        ----------
        REC_FILE_PATH : string acting as path to folder where received files are to be stored.           
        Scenario bandwidth and energy budget
        --------
        """
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.baseline_energy=configs['baseline_energy']
        self.baseline_tx=configs['base_tdl']
        self.energy_thresh=energy_budget
        self.energy_ratio=self.energy_thresh/self.baseline_energy
        self.t_UL=self.energy_ratio*self.baseline_tx
        self.total_data_sent=0
        self.throughputs=[]
        self.NEW_START=True
        config = get_string_config()
        self.faulty_data=os.path.join(config['file_paths']['test_files_dir'], config['file_paths']['faulty_data_file'])
        #Fetch only the "Allowed" part of the training data, according to the rules
        if self.NEW_START:
            make_initial_data(config['file_paths']['dataset_path'], config['file_paths']['test_files_dir'])
            with open(self.faulty_data, 'w') as f:
                f.write("")
        
        self.init_data=os.path.join(config['file_paths']['test_files_dir'], config['file_paths']['initial_data_file'])


        self.init_data_columns=pd.read_csv(self.init_data).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore').columns
        #Load the chosen model
        if config['ablation_settings']['use_DeepIoT']:
            from bin.DeepIoT_model import IoT_model
        else:
            from bin.IoT_model import IoT_model
        self.ml_model=IoT_model(self.init_data, 0.2)
        if self.NEW_START:
            #Train the initial model
            self.ml_model.train_initial_model()
        self.device_type="ES"
        #Load all configs to set up as TCP server.

        self.local_IP=configs['ESip']
        self.iot_device_PORT_TCP=configs['iot_device_PORT_TCP']
        self.iot_device_PORT_UDP=configs['iot_device_PORT_UDP']
        self.ESPORT=configs['ESPORT']
        self.rec_ip=configs['iot_device_ip']
        #Enable network control
        self.nc=network_control(self.device_type)
        if configs['use_config_network_control']==True:
            self.rate_kbps=bandwidth
            self.burst_kbps=16
            self.latency_ms=configs['buffering_latency_ms']
            self.packet_loss_pct=configs['packet_loss_pct']
            self.delay_ms=None
            self.jitter_ms=None
            self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
        iot_device_PORT=(self.iot_device_PORT_TCP, self.iot_device_PORT_UDP)
        self.file_Q=queue.Queue()
        self.configs=configs
        super().__init__(self.local_IP, self.ESPORT, self.rec_ip, iot_device_PORT, REC_FILE_PATH, self.device_type, self.file_Q)
    
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
        config = get_string_config()
        timestamps=pd.DataFrame(timestamps)
        timestamps.columns=[config['data_columns']['timestamp_column']]
        df2 = pd.concat([timestamps, data], axis=1).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
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
        config = get_string_config()
        init_data_no_faults=pd.read_csv(self.init_data).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        if os.path.getsize(init_data_path) <= 0:
            init_data=pd.DataFrame()
        else:
            init_data=pd.read_csv(init_data_path, on_bad_lines='skip').drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        timestamps=pd.DataFrame(timestamps)
        timestamps.columns=[config['data_columns']['timestamp_column']]
        df2 = pd.concat([timestamps, data], axis=1).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        df2.columns=init_data_no_faults.columns
        df_combined = pd.concat([init_data, df2], ignore_index=True).drop(columns=config['data_columns']['sensors_to_drop'], errors='ignore')
        df_combined.to_csv(init_data_path, index=False)

    def run(self):
        """
        Runs the basic server routine of waiting for samples, using them to improve the model, then transmitting the improved model back.

        Parameters:

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
        config = get_string_config()
        self.distribute_model(os.path.join(config['file_paths']['models_dir'], self.ml_model.model_name + config['file_extensions']['tflite_extension']))
        
        while Running:
            try:
                file, transmission_time = self.file_Q.get(timeout=3)
                if file=="DONE":
                    print("done")
                    print("Time elapsed: ", time.time()-start)
                    print("Transmitting time: ", self.time_transmitting)
                    print("Total data sent(KB): ", self.total_data_sent/1024)
                    print("TP transmissions: ", TP)
                    print("FP transmissions: ", FP)
                    remove_all_avro_files('received')
                    self.stop_TCP()
                    Running=False
                    #self.nc.reset_network_conditions()
                self.file_Q.task_done()
                if "ACK" in file:
                    config = get_string_config()
                    self.distribute_model(os.path.join(config['file_paths']['models_dir'], self.ml_model.model_name + config['file_extensions']['tflite_extension']))
                if config['file_extensions']['avro_extension'] in file:

                    data,timestamps, type, batch_num = AVRO.load_AVRO_file(file)
                    batches = np.array_split(data, batch_num)
                    for i, batch in enumerate(batches):
                        invert_training=False
                        if batch.iloc[:, -1].apply(lambda v: _label_matches(v, config['data_columns']['fault_label'])).any():
                            
                            print("INVERTED TRAINING")
                            invert_training=True
                            TP+=1
                        else:
                            FP+=1

                        self.ml_model.improve_model(batch.drop(batch.columns[-1], axis=1), invert_training, throughput=self.throughput, t_UL=self.t_UL)
                        self.throughputs.append(self.throughput)
                        if invert_training==False:
                            self.append_to_initial_data(data, timestamps, self.init_data)
                        else:
                            self.append_to_faulty_data(data, timestamps, self.faulty_data)
                    config = get_string_config()
                    self.distribute_model(os.path.join(config['file_paths']['models_dir'], self.ml_model.model_name + config['file_extensions']['tflite_extension']))
            except queue.Empty:
                pass
        return TP, FP, np.mean(self.throughputs)


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
        for ip in self.iot_device_devices:
            print("Sending model")
            self.send_file(ip, self.TAR_PORT_TCP,output_zip)
            

if __name__ == "__main__":
    
    es=ES_station("received", bandwidth=1000, energy_budget=30000)
    es.run()
