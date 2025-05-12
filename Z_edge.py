from TCP_code import TCP_COM
import time
import json
from network_control import network_control
from utils import make_dataset, generate_avro_schema, remove_all_avro_files
import pandas as pd
import zipfile
import numpy as np
import AVRO
import os
import queue
import shutil
import IoT_model
from alternative_iot_models import mlp_classifier
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import psutil
import threading
import csv
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

class Z_edge(TCP_COM):
    def __init__(self, REC_FILE_PATH, input):
        self.inference_batch=0
        self.use_PDR=False
        self.total_sent_data=0
        self.total_received_data=0
        self.num_inferences=0
        self.results=[]
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
            #rate_kbps=input
            rate_kbps=configs['bandwidth_limit_kbps']
            #burst_kbps=input
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
        self.fault_index=0
    
    def run(self):
        not_started=True
        while not_started:
            try:
                self.Ready_to_start()
                not_started=False
            except:
                pass
        try:
            while True:
                try:
                    file, rec_time= self.file_Q.get(timeout=2)
                    rssi=self.get_rssi_via_iw()
                    self.results.append((self.throughput, rssi))
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("Interrupted by user. Saving results to CSV...")
            with open("RSSI/throughputresults.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["throughput", "rssi"])  # Column headers
                writer.writerows(self.results)
            print("Results saved. Exiting.")

bs=Z_edge("received", 0.2)
bs.run()