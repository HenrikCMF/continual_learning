from TCP_code import TCP_COM
import time
import json
with open("configs.json", "r") as file:
    configs = json.load(file)
local_IP=configs['baseip']
PORT=configs['PORT']
rec_ip=configs['edgeip']
com_obj=TCP_COM(local_IP, PORT, rec_ip, PORT)

while True:
    time.sleep(0.1)