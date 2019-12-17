# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:42:09 2019

@author: Alero
"""

import cv2 as cv
import numpy as np

def sizing(img):
    img=cv.resize(img,(200,200))
    cv.imshow("ROI_THRESH",img)
    img=np.array(img,dtype=np.float32)
    img=img.reshape((1,200,200,1))
    
    return img