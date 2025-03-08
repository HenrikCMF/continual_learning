from TCP_code import TCP_COM
import json
import time
from network_control import network_control
class base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH):
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
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, edgePORT, REC_FILE_PATH, self.device_type)
    
    def receive_file(self, waittime=10):
        while True:
            time.sleep(waittime)
            #self.measure_PDR(100)
        #self.send_file("307.jpg")

    def distribute_model(self, model):
        for i in self.edge_devices:
            self.TAR_IP=i
            self.send_file(model)


bs=base_station("received")
#bs.send_file("307.jpg")
bs.receive_file(12)