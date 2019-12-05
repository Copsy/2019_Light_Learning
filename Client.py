# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 02:12:05 2019

@author: Alero
"""

WHITE=(255,255,255)
display_width=1000
display_height=200
window_size=(display_width,display_height)

TCP_IP="192.168.0.2"
TCP_PORT=12345

import socket
import cv2 as cv
import numpy as np
from camera_pipeline import pipeline
from Voice import speak

font=cv.FONT_HERSHEY_PLAIN
fontscale=5
font_location=(90,90)
zero=np.zeros((display_height,display_width), dtype=np.uint8)
x_1,y_1,x_2,y_2=100,100,340,340
SIZE=16

client_sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client_sock.connect((TCP_IP,TCP_PORT))
#cap=cv.VideoCapture(pipeline(framerate=60),cv.CAP_GSTREAM)
cap=cv.VideoCapture(0)

word=""

end_msg=str(-1)

while True:
    ref,frame=cap.read()
    frame=cv.flip(frame,1)
    cv.rectangle(frame,(x_1-5,y_1-5),(x_2+5,y_2+5),WHITE,3)
    
    cv.imshow("Figure_1", frame)
    
    cv.namedWindow("WORD")
    cv.moveWindow("WORD",300,550)
    cv.resizeWindow("WORD",window_size)
    cv.imshow("WORD",zero)
    
    key=cv.waitKey(30)
    if key==32:
        ROI=(frame.copy())[x_1:x_2,y_1:y_2]
        encode_param=[int(cv.IMWRITE_JPEG_QUALITY),90]
        result, encode=cv.imencode(".jpg", ROI, encode_param)
        data=np.array(encode)
        strData=data.tostring()
        #Send Image
        client_sock.send(str(len(strData)).ljust(16).encode())#b'len'
        client_sock.send(strData)
        # Waiting for Result
        try:
            msg=client_sock.recv(SIZE)
            word+=msg.decode()
        except Exception as e:
            _=0;

        cv.putText(zero,word,font_location,font,fontscale,WHITE,5)
        
    elif key==ord('a'):
        speak(word)
        
    elif key==ord('s'):#erase
        word=word[:-1]
        
    elif key==ord('d'):#RESET
        word=""
        zero=np.zeros((display_height,display_width), dtype=np.uint8)
        
    elif key==27:
        client_sock.send(end_msg.encode())
        break;

client_sock.close()
cap.release()
cv.destroyAllWindows()
