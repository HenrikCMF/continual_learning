import socket
import threading
import os
import struct
import time
from enum import Enum
import subprocess
import re
from utils import retry_transmission_handler, threaded
class telegram_type(Enum):
    PING=1
    DUMMY=2
    FILE=3

class TCP_COM():
    def __init__(self, MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT, REC_FILE_PATH, device):
        
        self.TAR_IP=TARGET_HOST
        if device=="edge":
            self.MY_PORT_TCP=MY_PORT[0]
            self.MY_PORT_UDP=MY_PORT[1]
            self.TAR_PORT_TCP=TARGET_PORT
        else:
            self.TAR_PORT_TCP=TARGET_PORT[0]
            self.TAR_PORT_UDP=TARGET_PORT[1]
            self.MY_PORT_TCP=MY_PORT
        self.__TCP_receive_file(MY_HOST, self.MY_PORT_TCP)
        self.in_path=REC_FILE_PATH
        self.device=device
        self.edge_devices=[]

    @retry_transmission_handler
    def send_file(self, client_socket, file_path):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f"Connected to {self.TAR_IP}:{self.TAR_PORT_TCP}")

        # Send file metadata
        client_socket.sendall(f"{telegram_type.FILE}:{file_name}:{file_size}".encode())
        ack = client_socket.recv(1024).decode()
        if ack != "READY":
            raise Exception("Not ready")
        # Send file content
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)


    @threaded
    def __TCP_receive_file(self,listen_host, listen_port):
        """Handles receiving files from the other party."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((listen_host, listen_port))
            server_socket.listen(5)
            print(f"Server listening on {listen_host}:{listen_port}")
            while True:

                conn, addr = server_socket.accept()
                if self.device=="edge":
                    if addr!=self.TAR_IP:
                        conn.close()
                        continue
                elif self.device=="bs":
                    if addr not in self.edge_devices:
                        self.edge_devices.append(addr[0])
                        print("added", addr[0])
                print(f"Connection established with {addr}")

                try:
                    # Receive file metadata
                    metadata = conn.recv(1024).decode()
                    type, file_name, file_size = metadata.split(":")
                    file_size = int(file_size)
                    conn.sendall("READY".encode())
                    if type==telegram_type.FILE:
                        # Receive file content
                        with open(os.path.join(self.in_path,f"received_{file_name}"), "wb") as f:
                            received_size = 0
                            while received_size < file_size:
                                data = conn.recv(1024)
                                if not data:
                                    break
                                f.write(data)
                                received_size += len(data)

                        print(f"File '{file_name}' received successfully!")
                except Exception as e:
                    pass

    
    def get_rssi_via_iw(self, interface="wlan0"):
        """Returns the current RSSI (dBm) for a given wireless interface, or None if unavailable."""
        try:
            cmd = ["iw", "dev", interface, "link"]
            output = subprocess.check_output(cmd, text=True).strip()
            
            # Example output line to parse might look like:
            #   "signal: -55 dBm"
            # Use a regex to find "signal: <value>"
            match = re.search(r"signal:\s+(-?\d+)\s+dBm", output)
            if match:
                return int(match.group(1))
            else:
                return None
        except subprocess.CalledProcessError:
            # Happens if the interface doesn't exist or there's an iw error.
            return None


if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)