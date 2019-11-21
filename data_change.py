'''

2019 Software Development

CODE BY LEE YU RYEOL

'''
import os
import numpy as np
import cv2 as cv

lookup = {} #---> " Label : Index
reverselookup = {} #---> Index : Label
count = 0
S_PATH="d:/working_direc/"
kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))

Cr_1=np.array([20,138,96])
Cr_2=np.array([235,169,128])
open_iteration=2
close_iteration=4

data_row=200
data_col=200
drop_rate=0.2

x_data=[]
y_data=[]
datacount=0

for j in os.listdir(S_PATH):#A to Z Folder
    lookup[j]=count
    reverselookup[count]=j
    count+=1


for i in range(0, count):
    for j in os.listdir(S_PATH+str(reverselookup[i])):
        origin_img=cv.imread(S_PATH+str(reverselookup[i])+"/"+str(j))
        origin_img=cv.GaussianBlur(origin_img,(3,3),0)
        img_ycc=cv.cvtColor(origin_img,cv.COLOR_BGR2YCrCb)
        mask_ycc=cv.inRange(img_ycc, Cr_1,Cr_2)
        ref, mask_ycc=cv.threshold(mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_CLOSE,kernel,iterations=close_iteration)
        mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_OPEN,kernel,iterations=open_iteration)
        img=cv.bitwise_and(origin_img,origin_img, mask=mask_ycc)
        
        cv.imwrite(S_PATH+str(reverselookup[i])+"/"+str(j),img)
        
        cv.imshow("Figure_1", img)
        key=cv.waitKey(10)
        if key==27:
            break;
    if key==27:
        break;
cv.destroyAllWindows()
