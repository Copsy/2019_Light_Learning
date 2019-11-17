import numpy as np
from keras.models import load_model
import cv2 as cv

'''
Leaning_Model_Test is CNN (N_Hidden_Layer-->Inside : Conv & Max_Padding)
Using ReLU & Adam Optimizer

'''
#Camera -> 640x480
"""

Setting ROI and that images is interperted change to alphabet by Using CNN
When "SpaceBar" button is pressed, that ROI is shoted

Author LEE YU RYEOL

"""



Cr_1=np.array([20,135,73])
Cr_2=np.array([255,190,158])


H5_path="./Learning_Model_V9.h5"

M_kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))
X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel=cv.getStructuringElement(cv.MORPH_RECT, (5,5))
x_1,y_1,x_2,y_2=40,140,280,360
target_string=""

model=load_model(H5_path)
Label=[]

test=np.zeros((200,200,1), dtype=np.float32)

test=test.reshape((1,200,200,1))

indexx=model.predict(test,verbose=0)

for i in range(len(indexx[0])):
    Label.append(chr(65+i))
    
del(test); del(indexx);

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
        img_ycc=cv.cvtColor(test_img, cv.COLOR_BGR2YCrCb)
        mask_ycc=cv.inRange(img_ycc, Cr_1, Cr_2)
        ref, mask_ycc=cv.threshold(mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_CLOSE,X_kernel,iterations=4)
        test_img=cv.bitwise_and(test_img,test_img, mask=mask_ycc)
        test_img=cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)#--->1Channel 200 x 200 x1
        test_img=cv.resize(test_img ,(200,200))
        cv.imshow("ROI_threshold", test_img)
        test_img=np.array(test_img, dtype=np.float32)
        test_img /= 255
        test_img=test_img.reshape((1,200,200,1))
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
