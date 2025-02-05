from TCP_code import TCP_COM
import json
with open("configs.json", "r") as file:
    configs = json.load(file)
local_IP=configs['baseip']
PORT=configs['PORT']
rec_ip=configs['edgeip']

com_obj=TCP_COM(local_IP, PORT, rec_ip, PORT)
com_obj.send_file("307.jpg")