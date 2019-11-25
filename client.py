import socket
import cv2 as cv
import numpy as np
from camera_pipeline import pipeline

TCP_IP="192.168.0.2"
TCP_PORT=5678
x_1,y_1,x_2,y_2=100,100,300,300
SIZE=8
client_sock=socket.socket()
client_sock.connect((TCP_IP,TCP_PORT))

cap=cv.VideoCapture(pipeline(flip_method=1), cv.CAP_GSTREAMER)

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
        break;

client_send("-1".encode())
client_sock.close()
cap.release()
cv.destroyAllWindows()
