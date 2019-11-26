# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:19:35 2019

@author: Alero
"""
import numpy as np
import os
from keras.models import load_model
import cv2 as cv
from Background_Abstract import abstract
from size_change import sizing
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
TCP_IP = '192.168.0.2'
TCP_PORT = 5678

ref=True

X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
H5_path="./Learning_Model_V9_1.h5"
model=load_model(H5_path)
Label=[]
count=0

for i in range(26):
    Label.append(chr(65+i))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("OPEN")
s.bind((TCP_IP, TCP_PORT))
print("Bind")
s.listen(True)
print("Listen")

while True:
    ref=True;    
    conn, addr = s.accept()
    print("Accept")
    conn.sendall((str(os.getpid())).encode())
    while True:         
        try:
            length, ref=recvall(conn,16)
            
            if ref==False:
                print("Close connection with client")
                break;
                
            stringData=recvall(conn,int(length))
            data=np.frombuffer(stringData,dtype="uint8")
            decimg=cv.imdecode(data,1)    
            
            img=cv.GaussianBlur(decimg,(3,3),0)
            img=cv.morphologyEx(img,cv.MORPH_CLOSE,X_kernel,iterations=1)
            img=cv.morphologyEx(img,cv.MORPH_OPEN, X_kernel,iterations=1)
            img=cv.flip(img,1)
            
            test_img=abstract(img)
            test_img=sizing(test_img)
            
            result=model.predict(test_img, verbose=0)
            index=np.argmax(result)
            conn.sendall(Label[index].encode())
            print(Label[index])
            cv.imshow("SERVER", decimg)
        except Exception as e:
            _=0;
        key=cv.waitKey(50)
        cv.destroyWindow("SERVER")
        if key==27:
            break;

s.close()
cv.destroyAllWindows()