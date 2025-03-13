import socket
import threading
import os
import struct
import time
from enum import Enum
import subprocess
import re
from utils import retry_transmission_handler, threaded
import random
class telegram_type(Enum):
    PDR=1
    DUMMY=2
    FILE=3

class TCP_COM():
    def __init__(self, MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT, REC_FILE_PATH, device, file_queue=None):
        self.MY_IP=MY_HOST
        self.TAR_IP=TARGET_HOST
        if device=="edge":
            self.MY_PORT_TCP=MY_PORT[0]
            self.MY_PORT_UDP=MY_PORT[1]
            self.TAR_PORT_TCP=TARGET_PORT
        else:
            self.TAR_PORT_TCP=TARGET_PORT[0]
            self.TAR_PORT_UDP=TARGET_PORT[1]
            self.MY_PORT_TCP=MY_PORT
        self.file_Q=file_queue
        self.__TCP_receive(MY_HOST, self.MY_PORT_TCP)
        self.in_path=REC_FILE_PATH
        self.device=device
        self.edge_devices=[]
        self.PDR=0
        if not os.path.exists(self.in_path):
            os.makedirs(self.in_path)

    @retry_transmission_handler
    def send_open_udp(self, client_socket, val=0, packet_num=0):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name=packet_num
        file_size=val
        client_socket.sendall(f"{telegram_type.PDR.value}:{file_name}:{file_size}".encode())

    @retry_transmission_handler
    def send_done_sending(self, client_socket, val=0, packet_num=0):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name="DONE"
        file_size=0
        client_socket.sendall(f"{telegram_type.DUMMY.value}:{file_name}:{file_size}".encode())
    
    def handle_dummy_req(self,file_size, file_name):
        file_size=float(file_size)
        if file_name=="DONE":
            self.file_Q.put(str("DONE"),0)
            print("Received done")


    @retry_transmission_handler
    def send_file(self, client_socket, file_path):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # Send file metadata
        client_socket.sendall(f"{telegram_type.FILE.value}:{file_name}:{file_size}".encode())
        ack = client_socket.recv(1024).decode()
        if ack != "READY":
            raise Exception("Not ready")
        # Send file content
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)

    def __receive_file(self, conn, file_name, file_size):
        conn.sendall("READY".encode())
        start=time.time()
        with open(os.path.join(self.in_path,f"r_{file_name}"), "wb") as f:
            received_size = 0
            while received_size < file_size:
                data = conn.recv(1024)
                if not data:
                    break
                f.write(data)
                received_size += len(data)
        stop=time.time()
        self.file_Q.put((str(os.path.join(self.in_path,f"r_{file_name}")),stop-start))
        print(f"File '{file_name}' received, took: ", stop-start)

    def handle_PDR_req(self, file_size, file_name):
        file_size=float(file_size)
        if file_size!=0:
            self.PDR=file_size
            print("RECEIVED PDR:", self.PDR)
        else:
            pdr=self.receive_packets(int(file_name))

    @threaded
    def __TCP_receive(self,listen_host, listen_port):
        """Handles receiving files from the other party."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((listen_host, listen_port))
            server_socket.listen(5)
            print(f"Server listening on {listen_host}:{listen_port}")
            while True:

                conn, addr = server_socket.accept()
                if self.device=="edge":
                    if addr[0]!=self.TAR_IP:
                        print("not target IP")
                        conn.close()
                        continue
                elif self.device=="bs":
                    if addr[0] not in self.edge_devices:
                        self.edge_devices.append(addr[0])
                        print("added", addr[0])
                try:
                    # Receive file metadata
                    metadata = conn.recv(1024).decode()
                    type, file_name, file_size = metadata.split(":")
                    print(" received:", type, file_name, file_size)
                    if telegram_type(int(type))==telegram_type.FILE:
                        file_size = int(file_size)
                        self.__receive_file(conn, file_name, file_size)
                    elif telegram_type(int(type))==telegram_type.PDR:
                        self.handle_PDR_req(file_size, file_name)
                    elif telegram_type(int(type))==telegram_type.DUMMY:
                        self.handle_dummy_req(file_size, file_name)
                except Exception as e:
                    print(e)

    def measure_PDR(self, num_packets):
        #Call edge device to listen for UDP packets
        print("PDR measure")
        self.send_open_udp(packet_num=num_packets)
        time.sleep(0.01)

        self.send_UDP_packets(num_packets=num_packets)

    def send_UDP_packets(self, num_packets=100, interval=0.001):
        """
        Sends `num_packets` UDP packets to (host, port) with a small interval between them.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(num_packets):
            message = f"{i}".encode()
            sock.sendto(message, (self.TAR_IP, self.TAR_PORT_UDP))
            time.sleep(interval)

        sock.close()
    
    def receive_packets(self, expected_packets=100):
        """
        Listens for UDP packets on (host, port) and counts how many arrive.
        """
        ENABLE_ARTIFICIAL_DROPS=False
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.MY_IP, self.MY_PORT_UDP))
        sock.settimeout(2)
        count = 0
        while count < expected_packets:
            try:
                data, addr = sock.recvfrom(1024)  # Block until a packet arrives
                if ENABLE_ARTIFICIAL_DROPS:
                    random_number = random.random()
                    if random_number<0.2:
                        data.decode()
                        continue
                if data:
                    count += 1
                    if count%20==0:
                        print(f"Received: {data.decode()} from {addr}")
            except:
                print("timed out")
                break

        sock.close()
        pdr = count / expected_packets
        print(f"PDR: {pdr*100:.2f}%")
        self.send_open_udp(val=pdr)
        return pdr

    #Replace with mmcli for cell connections
    def get_rssi_via_iw(self):
        """Returns the current RSSI (dBm) for a given wireless interface, or None if unavailable."""
        def get_wireless_interface():
            result = subprocess.run(["iwconfig"], capture_output=True, text=True)
            interfaces = re.findall(r"(\w+)\s+IEEE 802.11", result.stdout)
            return interfaces[0] if interfaces else None
        def get_signal_level(interface):
            cmd = f"iwconfig {interface} | grep 'Signal level'"
            try:
                output = subprocess.check_output(cmd, shell=True, text=True).strip()
                return output
            except subprocess.CalledProcessError:
                return None
        try:
            interface=get_wireless_interface()
            print("interface", interface)
            if interface:
                output=get_signal_level(interface)
                match = re.search(r"Signal level=(-?\d+)", output)
                return float(match.group(1))
            else:
                return None
        except subprocess.CalledProcessError:
            # Happens if the interface doesn't exist or there's an iw error.
            return None
#com

if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)