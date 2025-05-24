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
import warnings
from sklearn.exceptions import ConvergenceWarning
import subprocess
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
class Base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH, input):
        self.total_data_sent=0
        self.throughputs=[]
        self.use_PDR=False
        self.NEW_START=True
        self.faulty_data=os.path.join('test_files','faulty_data.csv')
        if self.NEW_START:
            make_initial_data("datasets/sensor.csv", 'test_files')
            with open(self.faulty_data, 'w') as f:
                f.write("")  # Write an empty string to create the file
        
        self.init_data=os.path.join('test_files','initial_data.csv')
        self.init_data_columns=pd.read_csv(self.init_data).drop(columns=["Unnamed: 0"], errors='ignore').columns
        self.ml_model=IoT_model.IoT_model(self.init_data, 0.2)
        #self.ml_model=mlp_classifier(self.init_data, input)
        if self.NEW_START:
            self.ml_model.train_initial_model()
        self.device_type="bs"
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['baseip']
        self.edgePORT_TCP=configs['edgePORT_TCP']
        self.edgePORT_UDP=configs['edgePORT_UDP']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['edgeip']
        self.nc=network_control(self.device_type)
        if configs['use_config_network_control']==True:
            self.rate_kbps=input
            self.burst_kbps=16#input
            #rate_kbps=configs['bandwidth_limit_kbps']
            #burst_kbps=configs['burst_limit_kbps']
            self.latency_ms=configs['buffering_latency_ms']
            self.packet_loss_pct=configs['packet_loss_pct']
            #delay_ms=configs['base_delay_ms']
            #jitter_ms=configs['jitter_ms']
            self.delay_ms=None
            self.jitter_ms=None
            self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        self.file_Q=queue.Queue()
        self.configs=configs
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, edgePORT, REC_FILE_PATH, self.device_type, self.file_Q)
    
    def append_to_initial_data(self, data, timestamps, init_data_path):
        #init_data=pd.read_csv(init_data_path).drop(columns=["Unnamed: 0"], errors='ignore')
        timestamps=pd.DataFrame(timestamps)
        timestamps.columns=['timestamp']
        df2 = pd.concat([timestamps, data], axis=1).drop(columns=["Unnamed: 0"], errors='ignore')
        #df2.columns=init_data.columns
        df2.columns=self.init_data_columns
        #df_combined = pd.concat([init_data, df2], ignore_index=True).drop(columns=["Unnamed: 0"], errors='ignore')
        #df_combined.to_csv(init_data_path)
        df2.to_csv(init_data_path, mode='a', header=False, index=False)

    def append_to_faulty_data(self, data, timestamps, init_data_path):
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

    def run(self, input):
        Running=True
        TP=0
        FP=0
        start=time.time()
        #Wait for X clients:
        clients=0
        while clients<1:
            file, transmission_time = self.file_Q.get(timeout=None, block=True)
            clients+=1
        if self.use_PDR:
            self.measure_PDR(100)
        self.distribute_model("models/"+self.ml_model.model_name+".tflite")
        
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
                    subprocess.run(f"sudo tc qdisc del dev {self.configs['baseNET_INTERFACE']} root", shell=True)
                self.file_Q.task_done()
                if "ACK" in file:
                    self.distribute_model("models/"+self.ml_model.model_name+".tflite")
                if ".avro" in file:
                    if self.use_PDR:
                        self.measure_PDR(100)
                    data,timestamps, type, batch_num = AVRO.load_AVRO_file(file)
                    batches = np.array_split(data, batch_num)
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
                    self.distribute_model("models/"+self.ml_model.model_name+".tflite")
                    #self.rate_kbps-=10
                    #self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
            except queue.Empty:
                #print("waiting for data")
                pass
        return TP, FP, np.mean(self.throughputs)
            #
        #self.send_file("307.jpg")

    def distribute_model(self, model):
        output_zip=model+'.zip'
        input_file=model
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(input_file, arcname=os.path.basename(input_file))
        self.total_data_sent+=os.path.getsize(output_zip)
        for ip in self.edge_devices:
            #self.TAR_IP=ip
            print("Sending model")
            self.send_file(ip, self.TAR_PORT_TCP,output_zip)
            
            #self.send_file(ip, self.TAR_PORT_TCP,model)
            #self.send_file(ip, self.TAR_PORT_TCP,"models/autoencoder.h5")

#bs=Base_station("received", 1000)
#bs.run(1000)