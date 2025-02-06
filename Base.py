from TCP_code import TCP_COM
import json
import time
class base_station(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['baseip']
        self.edgePORT=configs['edgePORT']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['edgeip']
        super().__init__(self.local_IP, self.basePORT, self.rec_ip, self.edgePORT, REC_FILE_PATH)
    
    def receive_file(self, waittime=10):
        time.sleep(waittime)
        #self.send_file("307.jpg")


bs=base_station("received")
bs.send_file("307.jpg")
bs.receive_file()