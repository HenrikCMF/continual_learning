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
class telegram_type(Enum):
    PDR=1
    DUMMY=2
    FILE=3

class TCP_COM():
    def __init__(self, MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT, REC_FILE_PATH, device, file_queue=None):
        self.throughput=None
        self.time_transmitting=0
        self.time_receiving=0
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
        self.RUNNING=True
        self.__TCP_receive(MY_HOST, self.MY_PORT_TCP)
        self.in_path=REC_FILE_PATH
        self.device=device
        self.edge_devices=[]
        self.PDR=0
        self.MSS=0
        if not os.path.exists(self.in_path):
            os.makedirs(self.in_path)

    @retry_transmission_handler
    def send_open_udp(self, client_socket, TAR_IP, val=0, packet_num=0):
        start=time.time()
        client_socket.connect((TAR_IP, self.TAR_PORT_TCP))
        file_name=packet_num
        file_size=val
        client_socket.sendall(f"{telegram_type.PDR.value}:{file_name}:{file_size}".encode())
        self.time_transmitting+=time.time()-start

    @retry_transmission_handler
    def send_done_sending(self, client_socket, val=0, packet_num=0):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name="DONE"
        file_size=0
        client_socket.sendall(f"{telegram_type.DUMMY.value}:{file_name}:{file_size}".encode())

    @retry_transmission_handler
    def send_ACK(self, client_socket, val=0, packet_num=0):
        start=time.time()
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name="ACK"
        file_size=0
        client_socket.sendall(f"{telegram_type.DUMMY.value}:{file_name}:{file_size}".encode())
        self.time_transmitting+=time.time()-start
    
    def handle_dummy_req(self,conn, file_size, file_name):
        file_size=float(file_size)
        if file_name=="DONE":
            self.file_Q.put((str("DONE"),0))
            print("Received done")
        elif file_name=="READY":
            conn.sendall("ACK".encode())
            self.file_Q.put((str("READY"),0))
        elif file_name=="ACK":
            conn.sendall("ACK".encode())
            self.file_Q.put((str("ACK"),0))
        elif file_name=="THROUGHPUT":
            conn.sendall("READY".encode())
            received_size = 0
            while received_size < file_size:
                data = conn.recv(1024)
                #print(received_size)
                if not data:
                    break
                received_size += len(data)
            conn.sendall("READY".encode())
        elif file_name=="PING":
            print("Received PING")
            conn.sendall("PONG".encode())
            print("sent back PONG")

    @retry_transmission_handler
    def Ready_to_start(self, client_socket, val=0, packet_num=0):
        client_socket.connect((self.TAR_IP, self.TAR_PORT_TCP))
        file_name="READY"
        file_size=0
        client_socket.sendall(f"{telegram_type.DUMMY.value}:{file_name}:{file_size}".encode())
        ack = client_socket.recv(1024).decode()
        if ack != "ACK":
            raise Exception("Not ready")
        else:
            self.file_Q.put((str("READY"),0))

    @retry_transmission_handler
    def send_file(self, client_socket, TAR_IP, TAR_PORT,file_path):
        
        start=time.time()
        client_socket.connect((TAR_IP, TAR_PORT))
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        client_socket.sendall(f"{telegram_type.FILE.value}:{file_name}:{file_size}".encode())
        ack = client_socket.recv(1024).decode()
        if ack != "READY":
            raise Exception("Not ready")
        # Send file content
        
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)
        ack = client_socket.recv(1024).decode()
        if ack != "DONE":
            raise Exception("Didnt finish")
        self.time_transmitting+=time.time()-start
        print("time of transmission:", time.time()-start)
            

    def __receive_file(self, conn, file_name, file_size):
        
        conn.sendall("READY".encode())
        start_time=time.perf_counter()
        with open(os.path.join(self.in_path,f"{file_name}"), "wb") as f:
            received_size = 0
            while received_size < file_size:
                data = conn.recv(1024)
                #print(received_size)
                if not data:
                    break
                f.write(data)
                received_size += len(data)
        conn.sendall("DONE".encode())
        stop_time=time.perf_counter()
        self.throughput = ((received_size+40) * 8) / ((stop_time - start_time-0.1) * 1000) #in kbps
        self.file_Q.put((str(os.path.join(self.in_path,f"{file_name}")),0))
        
        #print(f"File '{file_name}' received, took: ", stop-start)

    def handle_PDR_req(self, file_size, file_name):
        start=time.time()
        file_size=float(file_size)
        if file_size!=0:
            self.PDR=file_size
            print("RECEIVED PDR:", self.PDR)
        else:
            self.PDR=self.receive_packets(int(file_name))
        self.time_transmitting+=time.time()-start

    @threaded
    def __TCP_receive(self,listen_host, listen_port):
        """Handles receiving files from the other party."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512)
            server_socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_CONGESTION,b"bbr")
            server_socket.bind((listen_host, listen_port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            print(f"Server listening on {listen_host}:{listen_port}")
            while self.RUNNING:
                try:
                    conn, addr = server_socket.accept()
                    self.MSS = conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)
                    start=time.time()
                except socket.timeout:
                    continue
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
                        self.handle_dummy_req(conn,file_size, file_name)
                
                except Exception as e:
                    print(e)
                self.time_receiving+=time.time()-start
            print("stopped socket")
            server_socket.close()
    
    def stop_TCP(self):
        self.RUNNING=False


    def measure_PDR(self, num_packets):
        #Call edge device to listen for UDP packets
        for ip in self.edge_devices:
            print("PDR measure")
            self.send_open_udp(ip, packet_num=num_packets)
            time.sleep(0.01)
            self.send_UDP_packets(ip, num_packets=num_packets)

    def send_UDP_packets(self, ip, num_packets=100, interval=0.001):
        """
        Sends `num_packets` UDP packets to (host, port) with a small interval between them.
        """
        start=time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(num_packets):
            message = f"{i}".encode()
            sock.sendto(message, (ip, self.TAR_PORT_UDP))
            time.sleep(interval)

        sock.close()
        self.time_transmitting+=time.time()-start
    
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
        pdr = max(1-(count / expected_packets),0.001)
        print(f"PDR: {pdr*100:.2f}%")
        self.send_open_udp(self.TAR_IP,val=pdr)
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
        

    @retry_transmission_handler
    def getthroughput(self, client_socket, TAR_IP, TAR_PORT, data_size_bytes, RTT):
        try:
            # Connect and prepare to send
            client_socket.connect((TAR_IP, TAR_PORT))
            # Generate random data
            data = os.urandom(data_size_bytes)

            # Optional: Send a small header to inform receiver (e.g., data length)
            filename="THROUGHPUT"
            client_socket.sendall(f"{telegram_type.DUMMY.value}:{filename}:{data_size_bytes}".encode())
            #client_socket.sendall(f"DATA:{data_size_bytes}".encode())
            ack = client_socket.recv(1024).decode()
            if ack != "READY":
                raise Exception("Not ready")
            # Send data
            start_time = time.perf_counter()
            bytes_sent = 0
            chunk_size = 8192  # 1 KB
            for i in range(0, data_size_bytes, chunk_size):
                chunk = data[i:i+chunk_size]
                client_socket.sendall(chunk)
                bytes_sent += len(chunk)
            ack = client_socket.recv(1024).decode()
            if ack == "READY":
                end_time = time.perf_counter()
            # Measure results
            transmission_time = end_time - start_time  # seconds
            throughput_mbps = ((bytes_sent+20) * 8) / ((transmission_time-(RTT*2)) * 1000000)  # bits/sec to Mbps
            # Optionally store transmission time
            self.time_transmitting += transmission_time
            self.file_Q.put((throughput_mbps,0))
            return None

        except Exception as e:
            print(f"Error during data transfer: {e}")
            return None
    @retry_transmission_handler
    def measure_RTT(self, client_socket, TAR_IP, TAR_PORT):
        client_socket.connect((TAR_IP, TAR_PORT))
        start_rtt = time.perf_counter()
        filename="PING"
        size=len(filename)
        client_socket.sendall(f"{telegram_type.DUMMY.value}:{filename}:{size}".encode())
        pong = client_socket.recv(1024).decode()
        end_rtt = time.perf_counter()

        rtt = end_rtt - start_rtt  # seconds
        self.file_Q.put((rtt,0))

    def mathis_eq(self, RTT, PDR):
        if PDR<=0:
            PDR=0.001
        throughput=self.MSS/(RTT*np.sqrt(PDR))
        return throughput
#com

if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)