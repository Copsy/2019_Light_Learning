import socket
import numpy as np
import signal
import os


TCP_IP="172.18.57.83"
TCP_PORT=4088
SIZE=8
client_sock=socket.socket()
client_sock.connect((TCP_IP,TCP_PORT))
server_pid=client_sock.recv(SIZE).decode()
print("Server PID is "+server_pid)
while True:
    _=0;
    
client_sock.close()
