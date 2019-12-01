# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 02:12:05 2019

@author: Alero
"""

import socket
import numpy as np

UDP_IP="192.168.0.2"
UDP_PORT=12056
x_1,y_1,x_2,y_2=100,100,340,340
SIZE=10240

client_sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_sock.connect((UDP_IP,UDP_PORT))

client_sock.sendto("Hello world".encode(), (UDP_IP, UDP_PORT))


end_msg=str(-1)