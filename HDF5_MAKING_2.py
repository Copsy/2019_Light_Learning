# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:29:54 2019

@author: Lee Yu Ryeol
"""

import numpy as np
import h5py
import os
import cv2 as cv
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

F_PATH="d:/dataset_5/"
lookup={}
reverse_lookup={}
count=0
ran=0
image_number=0
width=200
height=200

rate_for_train=0.6

shuffle_data=True

label=[]
addr=[]

for j in os.listdir(F_PATH):
    lookup[j]=count
    reverse_lookup[count]=j
    count+=1

for i in range(len(reverse_lookup)):
    temp=glob.glob(F_PATH+reverse_lookup[i]+"/*.jpg")
    for j in range(len(temp)):
        addr.append(temp[j])
        label.append(i)
        
y_label=np.zeros((len(addr),count),dtype=np.uint8)    

for i in range(len(addr)):#Making one hot
    y_label[i][label[i]]=1
        
each_image_number=len(temp)
image_number=len(addr)
x_train, x_further, y_train,y_further=train_test_split(addr,y_label,test_size=0.4)
x_val, x_test, y_val, y_test=train_test_split(x_further,y_further, test_size=0.5)

del(temp); del(label); del(ran); del(j); del(i);


#y_label size (7,)
'''

x_train 31500 path
x_val   10500 path
x_test  10500 path

y_train 31500, 7
y_val   10500, 7
y_test  10500, 7

'''
    

f=h5py.File("./DATA_A_TO_G.hdf5", "w")

f.create_dataset("train_data",
                 shape=((int)(each_image_number*count*rate_for_train),width,height,1),
                 dtype=np.float32)
#52500 * 7 * 0.6
f.create_dataset("val_data",
                 shape=((int)(each_image_number*count*(1-rate_for_train)*0.5),
                        width,height,1),
                 dtype=np.float32)
f.create_dataset("test_data",
                 shape=((int)(each_image_number*count*(1-rate_for_train)*0.5),
                        width,height,1),
                 dtype=np.float32)

f.create_dataset("train_label",
                 shape=((int)(each_image_number*count*rate_for_train),count),
                 dtype=np.uint8)
# 7500 * 7 * 0.6
f.create_dataset("val_label",
                 shape=((int)(each_image_number*count*(1-rate_for_train)*0.5),count),
                 dtype=np.uint8)
# 7500 * 7 * 0.4 * 0.5 (Collect)
f.create_dataset("test_label",
                 shape=((int)(each_image_number*count*(1-rate_for_train)*0.5),count),
                 dtype=np.uint8)

print("Train_DATA_INPUT")
for i in range(len(x_train)):
    img=cv.imread(x_train[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["train_data"][i]=arr
    f["train_label"][i]=y_train[i]
    cv.imshow("Figure_1",img)
    keys=cv.waitKey(10)
    if keys==27:
        break;
cv.destroyAllWindows()

print("VAL_DATA_INPUT")
for i in range(len(x_val)):
    img=cv.imread(x_val[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["val_data"][i]=arr
    f["val_label"][i]=y_val[i]
    cv.imshow("Figure_1",img)
    keys=cv.waitKey(10)
    if keys==27:
        break;
cv.destroyAllWindows()
print("TEST_DATA_INPUT")
for i in range(len(x_test)):
    img=cv.imread(x_test[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["test_data"][i]=arr
    f["test_label"][i]=y_test[i]
    cv.imshow("Figure_1",img)
    keys=cv.waitKey(10)
    if keys==27:
        break;
cv.destroyAllWindows()



tier_1=list(f.keys())

f.close()