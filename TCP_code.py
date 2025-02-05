import socket
import threading
import os
class TCP_COM():
    def __init__(self, MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT):
        threading.Thread(target=self.receive_file, args=(MY_HOST, MY_PORT), daemon=True).start()
        self.TAR_IP=TARGET_HOST
        self.TAR_PORT=TARGET_PORT


    def send_file(self, file_path):
        """Handles sending files to the other party."""
        while True:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.TAR_IP, self.TAR_PORT))
                    print(f"Connected to {self.TAR_IP}:{self.TAR_PORT}")

                    # Send file metadata
                    client_socket.sendall(f"{file_name}:{file_size}".encode())
                    ack = client_socket.recv(1024).decode()
                    if ack != "READY":
                        print("Server not ready to receive. Aborting.")
                        continue

                    # Send file content
                    with open(file_path, "rb") as f:
                        while chunk := f.read(1024):
                            client_socket.sendall(chunk)

                    print("File sent successfully!")
            except Exception as e:
                print(f"Error while sending file: {e}")


    def receive_file(self,listen_host, listen_port):
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
                    file_name, file_size = metadata.split(":")
                    file_size = int(file_size)
                    conn.sendall("READY".encode())

                    # Receive file content
                    with open(f"received_{file_name}", "wb") as f:
                        received_size = 0
                        while received_size < file_size:
                            data = conn.recv(1024)
                            if not data:
                                break
                            f.write(data)
                            received_size += len(data)

                    print(f"File '{file_name}' received successfully!")
                except Exception as e:
                    print(f"Error while receiving file: {e}")
                finally:
                    conn.close()

if __name__ == "__main__":
    # Configure your local and target details
    MY_HOST = "0.0.0.0"   # Listen on all interfaces
    MY_PORT = 5000        # Port for receiving
    TARGET_HOST = "127.0.0.1"  # Change to target computer's IP
    TARGET_PORT = 6000        # Target port for sending
    COM_obj=TCP_COM(MY_HOST, MY_PORT, TARGET_HOST, TARGET_PORT)