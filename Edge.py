from TCP_code import TCP_COM
import time
import json
from network_control import network_control
from utils import make_dataset, generate_avro_schema
import pandas as pd
import numpy as np
import AVRO
import os
import queue
import shutil
import IoT_model
import matplotlib.pyplot as plt
class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        self.total_sent_data=0
        self.device_type="edge"
        self.model_path="models"
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['edgeip']
        self.edgePORT_TCP=configs['edgePORT_TCP']
        self.edgePORT_UDP=configs['edgePORT_UDP']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['baseip']
        self.nc=network_control(self.device_type)
        if configs['use_config_network_control']==True:
            rate_kbps=configs['bandwidth_limit_kbps']
            burst_kbps=configs['burst_limit_kbps']
            latency_ms=configs['buffering_latency_ms']
            packet_loss_pct=configs['packet_loss_pct']
            #delay_ms=configs['base_delay_ms']
            #jitter_ms=configs['jitter_ms']
            #packet_loss_pct=None
            delay_ms=None
            jitter_ms=None
            self.nc.set_network_conditions(rate_kbps, burst_kbps, latency_ms, packet_loss_pct, delay_ms, jitter_ms)
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        self.file_Q=queue.Queue()
        super().__init__(self.local_IP, edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH, self.device_type, self.file_Q)
        self.filename=make_dataset(1, 1)
        df=pd.read_csv(self.filename)
        self.timestamps=df['timestamp']
        self.data=df.drop(columns=['timestamp'])
        self.index=0
        sensors=np.shape(self.data)[1]
        self.len_of_dataset=np.shape(self.data)[0]
        self.schema_path="test_files/avro_"+str(sensors)+'.avsc'
        generate_avro_schema(sensors, self.schema_path)
        self.model = IoT_model.IoT_model("test_files/initial_data.csv")
        self.model.load_model()

    def run(self, waittime=10):
        sample_buffer=[]
        timestamp_buffer=[]
        mse_buff=[]
        done_sending=False
        while True:
            try:
                file, transmission_time= self.file_Q.get(timeout=0)
                print(file)
                if ".tflite" in file:
                    self.received_model(file)
                self.file_Q.task_done()
            except queue.Empty:
                if done_sending==False:
                    s, t=self.get_sample()
                    mse=self.model.calc_mse(s)
                    mse_buff.append(mse)
                    if self.index==self.len_of_dataset:
                        print("done")
                        plt.plot(mse_buff)
                        plt.show()
                    if mse>6:
                        print(t)
                        sample_buffer.append(s)
                        timestamp_buffer.append(t)
                    if len(timestamp_buffer)>5 or self.index==self.len_of_dataset: #Replace 5 with variable network parameter
                        sample_buffer=np.array(sample_buffer)
                        filename=os.path.join(
                            'test_files',
                            str(timestamp_buffer[0]).replace(" ", "-").replace(":", "-")+'.avro'
                            )
                        AVRO.save_AVRO_default(sample_buffer, timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=len(sample_buffer), codec='deflate')
                        self.total_sent_data+=os.path.getsize(filename)
                        self.send_file(filename)
                        sample_buffer=[]
                        timestamp_buffer=[]
                        if self.index==self.len_of_dataset:
                            done_sending=True
                            print("done")
                            plt.plot(mse_buff)
                            plt.show()
            except Exception as e:
                print(e)
        
    def received_model(self, path):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        destination_path=os.path.join(self.model_path, 'autoencoder.tflite')
        shutil.move(path, destination_path)
        self.model.load_model()
    
    def get_sample(self):
        #should fetch the next sample in the dataset
        sample=self.data.iloc[self.index]
        timestamp=self.timestamps.iloc[self.index]
        self.index+=1
        return sample, timestamp

#fd
bs=edge_device("received")
bs.run()
#bs.send_file("test_files/PEPE.jpeg")
#bs.receive_file()