from TCP_code import TCP_COM
import time
import json

class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['edgeip']
        self.PORT=configs['PORT']
        self.rec_ip=configs['baseip']
        super().__init__(self.local_IP, self.PORT, self.rec_ip, self.PORT, REC_FILE_PATH)

    def receive_file(self, waittime=10):
        time.sleep(waittime)
        self.send_file("307.jpg")


bs=edge_device("received")
bs.send_file("307.jpg")
bs.receive_file()