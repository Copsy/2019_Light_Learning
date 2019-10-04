# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 01:31:12 2019

Learning_Model using hdf5 file for input_data

@author: Lee Yu Ryeol
"""
from keras.utils.io_utils import HDF5Matrix
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import matplotlib.pylab as plt
import h5py

H5_PATH="./DATA_A_TO_G.hdf5"

ran=7
data_row=200; data_col=200;
epoch=50
batch_sizes=32
drop_rate=0.2
h5_f=h5py.File(H5_PATH,"r")
tier_1=list(h5_f.keys())
train_img_num=h5_f[tier_1[2]].shape[0]

x_train=HDF5Matrix(H5_PATH,str(tier_1[2]))
y_train=HDF5Matrix(H5_PATH,str(tier_1[3]))

x_val=HDF5Matrix(H5_PATH,str(tier_1[4]))
y_val=HDF5Matrix(H5_PATH,str(tier_1[5]))

x_test=HDF5Matrix(H5_PATH,str(tier_1[0]))
y_test=HDF5Matrix(H5_PATH,str(tier_1[1]))

model=models.Sequential(name="Model")

model.add(layers.Conv2D(32,(3,3),strides=(1,1),
                        padding="SAME",activation="relu",
                        input_shape=(x_train.shape[1:]),
                        kernel_initializer="glorot_normal",
                        kernel_regularizer=regularizers.l1_l2(l1=1e-1,l2=1e-2)))
model.add(layers.Conv2D(32,(3,3),strides=(1,1),
                       padding="SAME", activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
model.add(layers.Dropout(drop_rate))

model.add(layers.Conv2D(64,(3,3),strides=(1,1),
                        padding="SAME",activation="relu",
                        kernel_initializer="glorot_normal"))
model.add(layers.Conv2D(64,(3,3),strides=(1,1),
                       padding="SAME", activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
model.add(layers.Dropout(drop_rate))

model.add(layers.Conv2D(128,(3,3),strides=(1,1),
                        padding="SAME",activation="relu",
                        kernel_initializer="glorot_normal"))
model.add(layers.Conv2D(128,(3,3),strides=(1,1),
                       padding="SAME", activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
model.add(layers.Dropout(drop_rate))

model.add(layers.Flatten())
model.add(layers.BatchNormalization())

model.add(layers.Dense(128, activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.Dropout(drop_rate))

model.add(layers.Dense(128, activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.Dropout(drop_rate))

model.add(layers.Dense(64,activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.Dropout(drop_rate))

model.add(layers.Dense(32, activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.Dropout(drop_rate))


model.add(layers.Dense(ran, activation='softmax',
                       kernel_initializer="glorot_normal",
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2,l2=1e-2)))
model.add(layers.Dropout(drop_rate))

adam=optimizers.Adam(lr=5e-5,epsilon=1e-8)

model.compile(optimizer=adam,
              loss="categorical_hinge",
              metrics=["accuracy"])

hist=model.fit(x_train, y_train, batch_size=batch_sizes,
               epochs=epoch, verbose=1, shuffle="batch",
               validation_data=(x_val, y_val))

[loss,acc]=model.evaluate(x_test,y_test,verbose=1)
print("ACC : "+str(acc))

model.save("./Learning_Model_V4.h5",overwrite=True)
model.summary()

#To see Histroy of Training

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

h5_f.close()