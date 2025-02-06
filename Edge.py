from TCP_code import TCP_COM
import time
import json

class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['edgeip']
        self.edgePORT=configs['edgePORT']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['baseip']
        super().__init__(self.local_IP, self.edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH)

    def receive_file(self, waittime=10):
        time.sleep(waittime)
        self.send_file("307.jpg")

    def pdp_test(self):
        for i in range(20):
            self.meas_PDP()


bs=edge_device("received")
bs.pdp_test()
bs.send_file("307.jpg")
bs.receive_file()