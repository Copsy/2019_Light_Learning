# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:42:35 2019

@author: Alero
"""
import socket
import cv2
import numpy

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
        data=numpy.frombuffer(stringData,dtype="uint8")
        decimg=cv2.imdecode(data,1)
        cv2.imshow("SERVER", decimg)
        key=cv2.waitKey(1000)
        cv2.destroyWindow("SERVER")
        if key==27:
            break;

s.close()
cv2.destroyAllWindows()