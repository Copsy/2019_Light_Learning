# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:19:35 2019

@author: Alero
"""
import numpy as np
import socket

def recvall(sock, count):
    key=True
    buf = b''
    while count:
        newbuf = sock.recv(count)
        
        if len(newbuf)==2:
            print("Key is False")
            key=False
            return buf,key
            
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf, key 
'''
Need to change TCP_IP
'''
UDP_IP = "192.168.0.2"
UDP_PORT=12056
SIZE=40960
buffer = b''
ref=True


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("UDP_OPEN")
s.bind((UDP_IP, UDP_PORT))
print("Bind")

data=s.recv(1024)
buf=b''
buf+=data
aaa=np.frombuffer(buf,dtype="uint8")
print(aaa)

s.close()