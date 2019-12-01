# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:19:35 2019

@author: Alero
"""
import numpy as np
import os
from keras.models import load_model
import cv2 as cv
#배경 추출
from Background_Abstract import abstract
#이미지 크기 변경 200 200 1
from size_change import sizing
import socket

#스트링 바이트 스트림을 이미지로 변환
def recvall(sock, count):
    key=True
    buf = b''
    while count:
        newbuf = sock.recv(count)
        
        # 종료 신호의 확인
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
#클라이언트의 종료 확인 flag
ref=True

X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
# 저장된 모델을 불러 오기
H5_path="./Learning_Model_V9_1.h5"
model=load_model(H5_path)
Label=[]
count=0

#어디 알파벳 까지 구현되었는가
for i in range(26):
    Label.append(chr(65+i))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("OPEN")
s.bind((TCP_IP, TCP_PORT))
print("Bind")
s.listen(True)
print("Listen")
#통신 구간
while True:
    ref=True;    
    conn, addr = s.accept()
    print("Accept")
    while True:
        try:
            #이미지 올떄까지 기다림
            length, ref=recvall(conn,16)
            # 종료 신호 인지 이미지 인지 확인
            if ref==False:
                print("Close connection with client")
                break;
            # DEcode
            stringData, ref=recvall(conn,int(length))
            data=np.fromstring(stringData,dtype="uint8")
            decimg=cv.imdecode(data,1)
            
            # 이미지 전처리
            img=cv.GaussianBlur(decimg,(3,3),0)
            img=cv.morphologyEx(img,cv.MORPH_CLOSE,X_kernel,iterations=1)
            img=cv.morphologyEx(img,cv.MORPH_OPEN, X_kernel,iterations=1)
            img=cv.flip(img,1)
            
            test_img=abstract(img,1,4,2)
            test_img=sizing(test_img)
            #이미지 전처리 완료
            #평가 predict
            result=model.predict(test_img, verbose=0)
            index=np.argmax(result)
            #결과를 전송 ex) A 만 전송
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
