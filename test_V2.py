# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:50:56 2019

@author: Alero
"""

import numpy as np
from keras.models import load_model
import cv2 as cv
from Background_Abstract import abstract
from size_change import sizing

X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
H5_path="./Learning_Model_V9_1.h5"
M_kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))
x_1,y_1,x_2,y_2=40,140,285,365
target_string=""
model=load_model(H5_path)
Label=[]


for i in range(26):
    Label.append(chr(65+i))

cap=cv.VideoCapture(0)
count=0

while True:
    ref, origin_img=cap.read()
    img=origin_img
    #Rre-Processing Blur->Morphology_CLOSE ->Morphology_OPEN
    img=cv.GaussianBlur(origin_img,(3,3),0)
    img=cv.morphologyEx(img,cv.MORPH_CLOSE,X_kernel,iterations=1)
    img=cv.morphologyEx(img,cv.MORPH_OPEN, X_kernel,iterations=1)
    
    img=cv.flip(img,1)

    cv.rectangle(img,(x_1-5,y_1-5),(x_2+5,y_2+5),(255,255,255),3)
    
    ROI=(img.copy())[y_1:y_2, x_1:x_2]#Deep Copy
    
    key=cv.waitKey(1)
    
    cv.imshow("Figure_1", img)    

    if key==27:#ESC
        break;
    elif key==32: #SpaceBar
        count+=1
        test_img=ROI.copy()#Deep_Copy
        
        test_img=abstract(test_img,1,4,2)

        test_img=sizing(test_img)
        
        result=model.predict(test_img, verbose=0)
        
        index=np.argmax(result)
        print(Label[index])
        print("Count : " + str(count))
        target_string+=Label[index]
        
    elif key==ord('a'):
        
        print("String is : ",target_string)
        print(len(target_string))
        target_string=""
        
    elif key==ord('s'):
        
        target_string=target_string[:-1]
        print("Delete")
        
    elif key==ord('r'):
        
        target_string=""
        count=0
        print("Reset")

cv.destroyAllWindows()    
cap.release()