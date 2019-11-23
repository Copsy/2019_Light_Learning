# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:22:53 2019

@author: Alero
"""
import numpy as np
import cv2 as cv

def abstract(origin_image,mode,close_iteration, open_iteration):
    Cr_1=np.array([20,135,73])
    Cr_2=np.array([255,190,158])
    X_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])

    img_ycc=cv.cvtColor(origin_image,cv.COLOR_BGR2YCrCb)
    mask_ycc=cv.inRange(img_ycc, Cr_1,Cr_2)
    ref, mask_ycc=cv.threshold(mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_CLOSE,X_kernel,iterations=close_iteration)
    mask_ycc=cv.morphologyEx(mask_ycc,cv.MORPH_OPEN, X_kernel, iterations=open_iteration)
    img=cv.bitwise_and(origin_image,origin_image,mask=mask_ycc)
    
    if mode==1:#data_change into GRAY for predict
        img=cv.cvtColor(img ,cv.COLOR_BGR2GRAY)
        return img;
    
    return img;