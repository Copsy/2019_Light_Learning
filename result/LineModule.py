# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:56:50 2019

@author: Alero
"""

class LineModule:
    def __init__(self, drop_rate):
        self.H5_PATH="./DATA_A_TO_S.hdf5"

        self.drop_rate=drop_rate
        self.h5_f=h5py.File(self.H5_PATH,"r")
        self.tier_1=list(self.h5_f.keys())

        self.x_train=HDF5Matrix(self.H5_PATH,str(self.tier_1[2]))
        self.y_train=HDF5Matrix(self.H5_PATH,str(self.tier_1[3]))

        self.input_shape=layers.Input(shape=(self.x_train.shape[1:]))

        self.line_array = []
        
    def execute(self):
        lines=layers.Conv2D(32,(3,3),strides=(1,1),
                            padding="same")(self.input_shape)
        lines=layers.MaxPooling2D((2,2),strides=(2,2),padding="valid")(lines)
        lines=layers.BatchNormalization()(lines)


        lines=layers.Conv2D(32,(3,3),strides=(1,1),
                            padding="same")(lines)
        lines=layers.MaxPooling2D((2,2),strides=(2,2),padding="valid")(lines)
        lines=layers.BatchNormalization()(lines)


        lines=layers.Conv2D(64,(3,3),strides=(1,1),
                            padding="same")(lines)
        lines=layers.MaxPooling2D((2,2),strides=(2,2),padding="valid")(lines)
        lines=layers.BatchNormalization()(lines)

        lines=layers.Conv2D(64,(3,3),strides=(1,1),
                            padding="same")(lines)
        lines=layers.BatchNormalization()(lines)


        lines=layers.Flatten()(lines)

        lines=layers.BatchNormalization()(lines)
        lines=layers.Dense(512,activation="relu",
                        kernel_initializer="glorot_normal")(lines)
        lines=layers.Dropout(rate=self.drop_rate)(lines)


        lines=layers.BatchNormalization()(lines)
        lines=layers.Dense(32,activation="relu",
                        kernel_initializer="glorot_normal")(lines)
        lines=layers.Dropout(rate=self.drop_rate)(lines)

        self.line_array.append(lines)


    def result(self):
        return self.line_array
