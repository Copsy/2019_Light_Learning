'''

origin Input image size 200 200 1

Using Convolution & MaxPooling

To prevent Overfitting --> use dropout, regularizers
Use Adam For optimizer, loss --> cross entropy 

HardWare :
    CPU -> I7-8700K 3.7GHz~4.5GHz(Overclock)
    RAM -> 32GB (Overclock possible)
    GPU -> GTX-1070 (Overclock possible)
    
Using CUDA 10.1 in spyder3

'''
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.callbacks import TensorBoard
from time import time
import matplotlib.pylab as plt
import cv2 as cv
epoch=50
lookup = {} #---> " Label : Index
reverselookup = {} #---> Index : Label
count = 0
P_PATH="d:/dataset_5/" #P_PATH is path that has data

data_row=200
data_col=200
drop_rate=0.2

x_data=[]
y_data=[]
datacount=0
ran=0 # The number of Labels

for j in os.listdir(P_PATH):#A to Z Folder
    lookup[j]=count
    reverselookup[count]=j
    count+=1

ran=count

for i in range(0, ran):
    count=0
    tmp_y_value=[]
    for j in os.listdir(P_PATH+str(reverselookup[i])):
        # d:/dataset_3/00_A/asf.jpg
        img=cv.imread(P_PATH+str(reverselookup[i])+"/"+str(j))
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        arr=np.array(img,np.float32)
        arr/=255
        x_data.append(arr)
        count+=1
        
    tmp_y_value=np.full((count,1),i)
    y_data.append(tmp_y_value)
    datacount+=count
    

#Checking whether it is loaded normally
'''
for i in range(0, 29):
    plt.imshow(x_data[i*500,:,:])
    plt.title(reverselookup[y_data[i*3000,0]])
    plt.show()
'''

x_data=np.array(x_data)
y_data=np.array(y_data)
y_data=y_data.reshape((datacount,1))
y_data=to_categorical(y_data,dtype="uint8")#Check OK
x_data=x_data.reshape((datacount,data_row,data_col,1))

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.4)
#split-> For testing split 0.2 : 20%
#x_train : (16000, 120, 320, 1) / x_test : (2000,120,320, 1)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)


model=models.Sequential(name="Model")

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32,(3,3),strides=(1,1),
                        padding="SAME",activation="relu",
                        input_shape=(data_row,data_col,1),
                        kernel_initializer="glorot_normal",
                        kernel_regularizer=regularizers.l1_l2(l1=1e-2,l2=1e-2)))
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
#model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
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

model.add(layers.Dense(64,activation="relu",
                       kernel_initializer="glorot_normal"))
model.add(layers.Dropout(drop_rate))

model.add(layers.Dense(32, activation="relu",
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
rms_prop=optimizers.RMSprop(lr=1e-8, epsilon=1e-8)

#tensorboard=TensorBoard(log_dir="./logs/{}".format(time()))

model.compile(optimizer=adam,
              loss="categorical_hinge",
              metrics=["accuracy"])

'''
model.compile(optimizer=adam ,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
'''

hist=model.fit(x_train,y_train,
          epochs=epoch,
          batch_size=100,
          verbose=1,
          validation_data=(x_validate,y_validate))

#Verbose is status bar "1" that means Enable, and "2" is Disable
[loss,acc]=model.evaluate(x_test,y_test,verbose=1)
print("ACC : "+str(acc))

model.save("./Learning_Model_V3.h5",overwrite=True)
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
