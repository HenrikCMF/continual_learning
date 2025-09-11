import socket
import threading
import os
import struct
import time
from enum import Enum
import subprocess
import re
import numpy as np
from utils import retry_transmission_handler, threaded
import random
import queue
class telegram_type(Enum):
    PDR=1
    DUMMY=2
    FILE=3

class TCP_COM():
    def __init__(self, rec_queue, send_queue, REC_FILE_PATH, device, file_queue=None, throughput=1000):
        """
        Initializes TCP communication object, this object can be used for transmitting files or messages.
        Initializing it will also automatically spawn a TCP receiver thread based on the parameters given.

        Parameters:
        ----------
        MY_HOST : string giving device IP    
        MY_HOST : string giving device PORT
        TARGET_HOST : string giving target IP (IP of server if IoT, unused if server)   
        MY_HOST : string giving target PORT (port of server if IoT, port of IoT if server)  
        REC_FILE_PATH: string where received files will end up
        device: string either "edge" or "base" specifying whether IoT or server
        file_queue: python queue object to put messages for interthread communication.
        --------
        """
        self.throughput=throughput
        self.time_transmitting=0
        self.time_receiving=0
        self.recQ=rec_queue
        self.sendQ=send_queue
        self.file_Q=file_queue
        self.RUNNING=True
        self.__TCP_receive()
        self.in_path=REC_FILE_PATH
        self.device=device
        self.edge_devices=[]
        self.PDR=0
        self.MSS=0
        self.model_quantization=32
        if not os.path.exists(self.in_path):
            os.makedirs(self.in_path)


    def send_done_sending(self,val=0, packet_num=0):
        file_name="DONE"
        file_size=0
        self.sendQ.put((telegram_type.DUMMY, file_name, file_size, self.throughput, ""))

    def send_ACK(self, client_socket, val=0, packet_num=0):
        file_name="ACK"
        file_size=0
        self.sendQ.put((telegram_type.DUMMY, file_name, file_size, self.throughput, ""))
    
    def handle_dummy_req(self,file_size, file_name):
        if file_name=="DONE":
            self.file_Q.put((str("DONE"),0))
        elif file_name=="READY":
            self.file_Q.put((str("READY"),0))
        elif file_name=="ACK":
            self.file_Q.put((str("ACK"),0))

    def Ready_to_start(self, val=0, packet_num=0):
        file_name="READY"
        file_size=0
        self.sendQ.put((telegram_type.DUMMY, file_name, file_size, self.throughput, ""))
        self.file_Q.put((str("READY"),0))

    def send_file(self, payload, file_path):
        if self.model_quantization==8:
            file_name = "Q"+os.path.basename(file_path)
        else:
            file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        self.sendQ.put((telegram_type.FILE, file_name, file_size, self.throughput, payload))
        self.time_transmitting+=file_size/(self.throughput*1000/8)
        return file_size/(self.throughput*1000/8)
            

    def __receive_file(self, file_name, file_size, payload, t_time):
        if "Q" in file_name:
            self.model_quantization=8
        else:
            self.model_quantization=32
        if "tflite" in file_name:
            self.file_Q.put((str(file_name), t_time))
        else:
            self.file_Q.put((payload,t_time))
        
    def handle_telegram(self, telegram):
        #Transmission format: ((type, file_name, file_size, throughput, payload))
        type, file_name, file_size, throughput, payload = telegram
        self.time_receiving+=file_size/(throughput*1000/8)
        self.throughput=throughput
        #print(self.device," received: ", type, file_name, file_size)
        if type==telegram_type.FILE:
            file_size = int(file_size)
            self.__receive_file(file_name, file_size, payload, file_size/(throughput*1000/8))
        elif type==telegram_type.DUMMY:
            self.handle_dummy_req(file_size, file_name)

    @threaded
    def __TCP_receive(self):
        """Handles receiving files from the other party."""
        while self.RUNNING:
            try:
                telegram = self.recQ.get_nowait()
                self.handle_telegram(telegram)
                self.recQ.task_done()
            except queue.Empty:
                time.sleep(0.1)

    def stop_TCP(self):
        self.RUNNING=False


    

    
        


#com

if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)