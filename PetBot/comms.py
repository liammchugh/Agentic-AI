import socket
import threading
import time
import random

# Constants
HOST = '192.168.1.1'  # Replace with the IP address of the Raspberry Pi
PORT = 5000
DATA_CHANNELS = 50
INFO_PACKET_DIMENSIONS = 25

# Function to send data channels
def send_data_channels(sock):
    while True:
        data = [random.random() for _ in range(DATA_CHANNELS)]
        message = ','.join(map(str, data))
        sock.sendall(message.encode('utf-8'))
        time.sleep(0.1)  # Adjust the frequency as needed

# Function to receive info packets
def receive_info_packets(sock):
    while True:
        data = sock.recv(1024).decode('utf-8')
        if data:
            info_packet = list(map(float, data.split(',')))
            if len(info_packet) == INFO_PACKET_DIMENSIONS:
                print(f"Received info packet: {info_packet}")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        
        # Start sending data channels
        send_thread = threading.Thread(target=send_data_channels, args=(sock,))
        send_thread.start()
        
        # Start receiving info packets
        receive_thread = threading.Thread(target=receive_info_packets, args=(sock,))
        receive_thread.start()
        
        # Keep the main thread alive
        send_thread.join()
        receive_thread.join()

if __name__ == "__main__":
    main()