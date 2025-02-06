import socket
import threading
import os
import struct
import time
from enum import Enum
from utils import retry_until_success, threaded
class telegram_type(Enum):
    PING=1
    DUMMY=2
    FILE=3

class TCP_COM():
    def __init__(self, MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT, REC_FILE_PATH):
        #threading.Thread(target=self.__receive_file, args=(MY_HOST, MY_PORT), daemon=True).start()
        self.__receive_file(MY_HOST, MY_PORT)
        self.TAR_IP=TARGET_HOST
        self.TAR_PORT=TARGET_PORT
        self.in_path=REC_FILE_PATH

    @retry_until_success
    def send_file(self, client_socket, file_path):
        client_socket.connect((self.TAR_IP, self.TAR_PORT))
        print(2)
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f"Connected to {self.TAR_IP}:{self.TAR_PORT}")

        # Send file metadata
        client_socket.sendall(f"{telegram_type.FILE}:{file_name}:{file_size}".encode())
        print(3)
        ack = client_socket.recv(1024).decode()
        if ack != "READY":
            raise Exception("Not ready")
        print(4)
        # Send file content
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)


    @threaded
    def __receive_file(self,listen_host, listen_port):
        """Handles receiving files from the other party."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((listen_host, listen_port))
            server_socket.listen(5)
            print(f"Server listening on {listen_host}:{listen_port}")

            while True:
                conn, addr = server_socket.accept()
                print(f"Connection established with {addr}")

                try:
                    # Receive file metadata
                    metadata = conn.recv(1024).decode()
                    type, file_name, file_size = metadata.split(":")
                    file_size = int(file_size)
                    conn.sendall("READY".encode())

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

    def meas_PDP(self):

        def get_tcp_info(sock):
            # The tcp_info struct layout may vary by kernel version.
            # Here we assume a common layout: 7 unsigned chars followed by 21 unsigned ints.
            # Total size expected = 7 + 21*4 = 91 bytes (it might be padded to 104 bytes, so we request 104).
            fmt = "B" * 7 + "I" * 21
            buf = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
            return struct.unpack(fmt, buf)
        s = socket.create_connection((self.TAR_IP, self.TAR_PORT))
        # Send minimal data to trigger some activity
        s.send(b"Yo Homie, its ya boi Tony")

        # Give it a little time to process the request and possibly retransmit if needed.
        time.sleep(0.5)

        info = get_tcp_info(s)

        # NOTE: The index for tcpi_total_retrans may vary.
        # In many kernels, it is the 19th element (index 18) in the unpacked tuple.
        tcpi_total_retrans = info[18]
        print("32-35", info[32],info[33],info[34],info[35],)
        print("Total retransmissions:", tcpi_total_retrans)
        s.close()

if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)