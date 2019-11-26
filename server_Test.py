# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:19:35 2019

@author: Alero
"""
import numpy as np
import socket
from threading import Thread
import signal
import os

def sigpipe_handler(signum, frame):
    print("Hello wolrd")
    
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
#SIGALRM is 14
signal.signal(signal.SIGALRM, sigpipe_handler)

'''
Need to change TCP_IP
'''
TCP_IP = '172.18.57.83'
TCP_PORT = 4088

print("My PID is "+str(os.getpid()))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("OPEN")
s.bind((TCP_IP, TCP_PORT))
print("Bind")
s.listen(True)
print("Listen")

while True:    
    conn, addr = s.accept()
    print("Accept")
    conn.sendall((str(os.getpid())).encode())
    while True:         
        try:
            length=recvall(conn,16)
            print("Length is "+length)
            stringData=recvall(conn,int(length))
            print("String Data is "+stringData)
            data=np.frombuffer(stringData,dtype="uint8")
            print("Data is "+data)
        except Exception as e:
            _=0
s.close()
cv.destroyAllWindows()
