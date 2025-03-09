from TCP_code import TCP_COM
import time
import json
from network_control import network_control
from utils import make_dataset, generate_avro_schema
import pandas as pd
import numpy as np
import AVRO
import os
from queue import Queue
class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        self.device_type="edge"
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
        self.file_Q=Queue()
        super().__init__(self.local_IP, edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH, self.device_type, self.file_Q)
        self.filename=make_dataset(1, 1)
        df=pd.read_csv(self.filename)
        self.timestamps=df['timestamp']
        self.data=df.drop(columns=['timestamp'])
        print(self.timestamps.head())
        print(self.data.head())
        self.index=0
        sensors=np.shape(self.data)[1]
        self.len_of_dataset=np.shape(self.data)[0]
        self.schema_path="test_files/avro_"+str(sensors)+'.avsc'
        generate_avro_schema(sensors, self.schema_path)

    def run(self, waittime=10):
        sample_buffer=[]
        timestamp_buffer=[]
        while True:
            s, t=self.get_sample()
            if self.index%10000==0:
                sample_buffer.append(s)
                timestamp_buffer.append(t)
            if len(timestamp_buffer)>5 or self.index==self.len_of_dataset:
                sample_buffer=np.array(sample_buffer)
                filename=os.path.join(
                    'test_files',
                    str(timestamp_buffer[0]).replace(" ", "-").replace(":", "-")+'.avro'
                    )
                print(timestamp_buffer)
                AVRO.save_AVRO_default(sample_buffer, timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=len(sample_buffer), codec='deflate')
                self.send_file(filename)
                sample_buffer=[]
                timestamp_buffer=[]
                if self.index==self.len_of_dataset:
                    exit()

    
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