from TCP_code import TCP_COM
import json
import time
class base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['baseip']
        self.edgePORT_TCP=configs['edgePORT_TCP']
        self.edgePORT_UDP=configs['edgePORT_UDP']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['edgeip']
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, edgePORT, REC_FILE_PATH, "bs")
    
    def receive_file(self, waittime=10):
        time.sleep(waittime)
        #self.send_file("307.jpg")

    def distribute_model(self, model):
        for i in self.edge_devices:
            self.TAR_IP=i
            self.send_file(model)


bs=base_station("received")
#bs.send_file("307.jpg")
bs.receive_file()