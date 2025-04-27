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

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
class Z_Base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH, input):
        self.total_data_sent=0
        self.use_PDR=False
        self.NEW_START=True
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
            #rate_kbps=input
            rate_kbps=configs['bandwidth_limit_kbps']
            #burst_kbps=input
            burst_kbps=configs['burst_limit_kbps']
            latency_ms=configs['buffering_latency_ms']
            packet_loss_pct=configs['packet_loss_pct']
            #delay_ms=configs['base_delay_ms']
            #jitter_ms=configs['jitter_ms']
            delay_ms=None
            jitter_ms=None
            self.nc.set_network_conditions(rate_kbps, burst_kbps, latency_ms, packet_loss_pct, delay_ms, jitter_ms)
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        self.file_Q=queue.Queue()
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, edgePORT, REC_FILE_PATH, self.device_type, self.file_Q)


    def run(self):
        clients=0
        while clients<1:
            file, transmission_time = self.file_Q.get(timeout=None, block=True)
            clients+=1
        for i in range(50):
            for ip in self.edge_devices:
                #self.TAR_IP=ip
                self.measure_RTT(ip, self.TAR_PORT_TCP)
                try:
                    file, transmission_time = self.file_Q.get(timeout=3)
                    RTT=file
                except queue.Empty:
                    #print("waiting for data")
                    pass
                self.getthroughput(ip, self.TAR_PORT_TCP, 10000, RTT)
                try:
                    file, transmission_time = self.file_Q.get(timeout=3)
                    print("through",file)
                except queue.Empty:
                    #print("waiting for data")
                    pass

bs=Z_Base_station("received", 0.2)
bs.run()