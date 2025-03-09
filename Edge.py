from TCP_code import TCP_COM
import time
import json
from network_control import network_control
from utils import make_dataset, generate_avro_schema
import pandas as pd
import numpy as np
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
        super().__init__(self.local_IP, edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH, self.device_type)
        self.filename=make_dataset(1, 1)
        df=pd.read_csv(self.filename)
        self.timestamps=df['timestamp']
        self.data=df.drop(columns=['timestamp'])
        print(self.timestamps.head())
        print(self.data.head())
        self.index=0
        sensors=np.shape(self.data)[1]
        generate_avro_schema(sensors, "testfiles_avro_"+str(sensors)+'.avsc')

    def receive_file(self, waittime=10):
        while True:
            time.sleep(waittime)
            #print("RSSI",self.get_rssi_via_iw())
            #self.send_file("test_files/PEPE.jpeg")
            s=self.get_sample()
    
    def get_sample(self):
        #should fetch the next sample in the dataset
        sample=self.data.iloc[self.index]
        self.index+=1
        print(sample)
        return sample

#fd
bs=edge_device("received")
bs.send_file("test_files/PEPE.jpeg")
bs.receive_file()