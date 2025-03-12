from TCP_code import TCP_COM
import json
import time
import queue
from network_control import network_control
import IoT_model
import AVRO
import os
from utils import make_initial_data
import numpy as np
import pandas as pd
class base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        self.total_data_sent=0
        self.NEW_START=True
        if self.NEW_START:
            make_initial_data("datasets/sensor.csv", 'test_files')
        self.init_data=os.path.join('test_files','initial_data.csv')
        ml_model=IoT_model.IoT_model(self.init_data)
        if self.NEW_START:
            ml_model.train_initial_model()
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
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, edgePORT, REC_FILE_PATH, self.device_type, self.file_Q)
    
    def append_to_initial_data(self, data, timestamps, init_data_path):
        init_data=pd.read_csv(init_data_path)
        timestamps=pd.DataFrame(timestamps)
        df2 = pd.concat([timestamps, data], axis=1)
        df2.columns=init_data.columns
        df_combined = pd.concat([init_data, df2], ignore_index=True)
        df_combined.to_csv(init_data_path)
        print("updated CSV")

    def receive_file(self, waittime=10):
        while True:
            try:
                file, transmission_time = self.file_Q.get(timeout=3)
                self.file_Q.task_done()
                time.sleep(waittime)
                data,timestamps, type, metadata = AVRO.load_AVRO_file(file)
                self.append_to_initial_data(data, timestamps, self.init_data)
                #self.distribute_model("models/autoencoder.tflite")
            except queue.Empty:
                pass
            #self.measure_PDR(100)
        #self.send_file("307.jpg")

    def distribute_model(self, model):
        for i in self.edge_devices:
            self.TAR_IP=i
            self.send_file(model)


bs=base_station("received")
#bs.send_file("307.jpg")
bs.receive_file(12)