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

F_PATH="d:/dataset_5/"
reverse_lookup={}
count=0
ran=0
width=200
height=200

# 60% 학습비율
rate_for_train=0.6

#Randomize
shuffle_data=True

#Label 알파벳
label=[]
# addr은 파일 경로
addr=[]

#파일 위치 확인
for j in os.listdir(F_PATH):
    reverse_lookup[count]=j
    count+=1

#파일 Load
for i in range(len(reverse_lookup)):
    temp=glob.glob(F_PATH+reverse_lookup[i]+"/*.jpg")
    for j in range(len(temp)):
        addr.append(temp[j])
        label.append(i)
        
#One hot : 원하는 알파벳을 0, 1로 이루어진 list로 표현 b : 0100000.....
y_label=np.zeros((len(addr),count),dtype=np.uint8)    

for i in range(len(addr)):#Making one hot
    y_label[i][label[i]]=1

#알파벳 당의 파일 갯수
each_image_number=len(temp)
#분할 학습용/ Valid/ Test
x_train, x_further, y_train,y_further=train_test_split(addr,y_label,
                                                       test_size=(1-rate_for_train))
x_val, x_test, y_val, y_test=train_test_split(x_further,y_further, test_size=0.5)

del(temp); del(label); del(ran); del(j); del(i); del(reverse_lookup);

#HDF5 Database 생성
f=h5py.File("d:/aaa/DATA_A_TO_Z.hdf5", "w")

# Node 생성
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

# Insert Data to node
print("Train_DATA_INPUT")
for i in range(len(x_train)):
    img=cv.imread(x_train[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["train_data"][i]=arr
    f["train_label"][i]=y_train[i]
    print("Train_Input : "+str(i))

print("VAL_DATA_INPUT")
for i in range(len(x_val)):
    img=cv.imread(x_val[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["val_data"][i]=arr
    f["val_label"][i]=y_val[i]
    print("Valid_Input : "+str(i))

print("TEST_DATA_INPUT")
for i in range(len(x_test)):
    img=cv.imread(x_test[i])
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arr=np.array(img,np.float32)
    arr=np.resize(arr,(200,200,1))
    arr/=255
    f["test_data"][i]=arr
    f["test_label"][i]=y_test[i]
    print("Test_Input : "+str(i))
    
tier_1=list(f.keys())

f.close()
