# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 02:12:05 2019

@author: Alero
"""

import socket
import cv2 as cv
import numpy as np

TCP_IP="192.168.0.2"
TCP_PORT=5678
x_1,y_1,x_2,y_2=100,100,300,300
SIZE=8
client_sock=socket.socket()
client_sock.connect((TCP_IP,TCP_PORT))
server_pid=client_sock.recv(SIZE).decode()
print("Server PID is "+server_pid)

cap=cv.VideoCapture(0)

end_msg=str(-1)

while True:
    ref,frame=cap.read()
    
    cv.rectangle(frame,(x_1-5,y_1-5),(x_2+5,y_2+5),(255,255,255),3)
    
    cv.imshow("Figure_1", frame)
    key=cv.waitKey(30)
    if key==32:
        ROI=(frame.copy())[x_1:x_2,y_1:y_2]
        encode_param=[int(cv.IMWRITE_JPEG_QUALITY),90]
        result, encode=cv.imencode(".jpg", ROI, encode_param)
        data=np.array(encode)
        strData=data.tostring()
        #Send Image
        client_sock.send(str(len(strData)).ljust(16).encode())
        client_sock.send(strData)
        # Waiting for Result
        try:
            msg=client_sock.recv(SIZE)
        except Exception as e:
            _=0;
        print("Result is "+msg.decode())
    elif key==27:
        client_sock.send(end_msg.encode())
        break;

client_sock.close()
cap.release()
cv.destroyAllWindows()