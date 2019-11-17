from keras.utils.io_utils import HDF5Matrix
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import matplotlib.pylab as plt
import h5py
from . import LineModule

H5_PATH="./DATA_A_TO_S.hdf5"

epoch=20
batch_sizes=64
drop_rate=0.4
h5_f=h5py.File(H5_PATH,"r")
tier_1=list(h5_f.keys())

x_train=HDF5Matrix(H5_PATH,str(tier_1[2]))
y_train=HDF5Matrix(H5_PATH,str(tier_1[3]))
# from val set, there is shuffled
x_val=HDF5Matrix(H5_PATH,str(tier_1[4]))
y_val=HDF5Matrix(H5_PATH,str(tier_1[5]))

x_test=HDF5Matrix(H5_PATH,str(tier_1[0]))
y_test=HDF5Matrix(H5_PATH,str(tier_1[1]))

ran=y_train.shape[1] 

newmodel = LineModule(drop_rate)
line_1 = newmodel.execute()
line_2 = newmodel.execute()
line_3 = newmodel.execute()
results = newmodel.result()

input_shape=newmodel.input_shape

x1=layers.concatenate(results)

output_shape=layers.Dense(ran,activation="softmax",
                        kernel_initializer="glorot_normal",
                        kernel_regularizer=regularizers.l1_l2(l1=1e-2,l2=1e-2))(x1)

model=models.Model(inputs=input_shape,outputs=output_shape)

adam=optimizers.Adam(lr=1e-5)

model.compile(optimizer=adam,
            loss="categorical_hinge",
            metrics=["accuracy"])



hist=model.fit(x_train, y_train, batch_size=batch_sizes,
            epochs=epoch, verbose=1, shuffle="batch",
            validation_data=(x_val, y_val)) 

[loss,accuracy]=model.evaluate(x_test,y_test,verbose=1)
print("ACC : "+str(accuracy))

model.save("./Learning_Model_V9.h5",overwrite=True)
model.summary()

h5_f.close()

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
