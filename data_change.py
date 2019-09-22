import os
import numpy as np
import cv2 as cv

lookup = {} #---> " Label : Index
reverselookup = {} #---> Index : Label
count = 0
P_PATH="d:/dataset/" #P_PATH is path that has data
S_PATH="d:/dataset_5/"
kernel=cv.getStructuringElement(cv.MORPH_RECT, (5,5))

step=0

Cr_1=np.array([40,135,73])
Cr_2=np.array([220,160,138])

data_row=200
data_col=200
drop_rate=0.2

x_data=[]
y_data=[]
datacount=0
ran=0 # The number of Labels

for j in os.listdir(S_PATH):#A to Z Folder
    lookup[j]=count
    reverselookup[count]=j
    count+=1

ran=count

for i in range(0, ran):
    step=0
    for j in os.listdir(S_PATH+str(reverselookup[i])):
        origin_img=cv.imread(S_PATH+str(reverselookup[i])+"/"+str(j))
        img_ycc=cv.cvtColor(origin_img,cv.COLOR_BGR2YCrCb)
        mask_ycc=cv.inRange(img_ycc, Cr_1,Cr_2)
        ref, mask_ycc=cv.threshold(mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_CLOSE,kernel,iterations=2)
        img=cv.bitwise_and(origin_img,origin_img, mask=mask_ycc)
        
        cv.imwrite(S_PATH+str(reverselookup[i])+"/"+str(j),img)
        
        cv.imshow("Figure_1", img)
        key=cv.waitKey(10)
        if key==27:
            break;
    if key==27:
        break;
cv.destroyAllWindows()
