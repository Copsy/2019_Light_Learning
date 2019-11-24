# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:19:35 2019

@author: Alero
"""
import numpy as np
from keras.models import load_model
import cv2 as cv
from Background_Abstract import abstract
from size_change import sizing
import socket

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = '192.168.0.2'
TCP_PORT = 5678

X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
H5_path="./Learning_Model_V9_1.h5"
model=load_model(H5_path)
Label=[]
count=0

for i in range(26):
    Label.append(chr(65+i))


#TCP소켓 열고 수신 대기
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("OPEN")
s.bind((TCP_IP, TCP_PORT))
print("Bind")
s.listen(True)
print("Listen")

while True:    
    conn, addr = s.accept()
    print("Accept")
    while True:
        length = recvall(conn,16) #Call function
        stringData = recvall(conn, int(length))
        #stringData = recvall(conn, int(length))
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
        
        print(Label[index])
        
        cv.imshow("SERVER", decimg)
        key=cv.waitKey(1000)
        cv.destroyWindow("SERVER")
        if key==27:
            break;

s.close()
cv.destroyAllWindows()