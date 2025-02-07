from TCP_code import TCP_COM
import time
import json

class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['edgeip']
        self.edgePORT_TCP=configs['edgePORT_TCP']
        self.edgePORT_UDP=configs['edgePORT_UDP']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['baseip']
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        super().__init__(self.local_IP, edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH, "edge")

    def receive_file(self, waittime=10):
        time.sleep(waittime)
        self.send_file("307.jpg")

    def pdp_test(self):
        for i in range(20):
            self.meas_PDP()


bs=edge_device("received")
bs.send_file("307.jpg")
bs.receive_file()