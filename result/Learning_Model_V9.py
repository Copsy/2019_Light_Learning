from keras.utils.io_utils import HDF5Matrix
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import matplotlib.pylab as plt
import h5py
from . import LineModule

H5_PATH="./DATA_A_TO_S.hdf5"



#파라미터 불러오기



epoch=20
batch_sizes=64
drop_rate=0.4
# PIPELINE 구축을 위한 H5PY 파일 로드
h5_f=h5py.File(H5_PATH,"r")
#H5 파일(h5_f)은 Dictionary 파일이므로 Value를 찾기위해 Key를 Load한다
tier_1=list(h5_f.keys())


# 로드한 키를 이용하여 이미지 파일을 링크한다
x_train=HDF5Matrix(H5_PATH,str(tier_1[2]))
# 로드한 키를 이용하여 Label을 링크한다
y_train=HDF5Matrix(H5_PATH,str(tier_1[3]))
# from val set, there is shuffled
x_val=HDF5Matrix(H5_PATH,str(tier_1[4]))
y_val=HDF5Matrix(H5_PATH,str(tier_1[5]))

x_test=HDF5Matrix(H5_PATH,str(tier_1[0]))
y_test=HDF5Matrix(H5_PATH,str(tier_1[1]))

# Label 갯수 불러오기 Output Layer의 Softmax 에서 이용되어짐
ran=y_train.shape[1] 

# 모델 구축
newmodel = LineModule(drop_rate)
line_1 = newmodel.execute()
line_2 = newmodel.execute()
line_3 = newmodel.execute()
results = newmodel.result()

#모델 구축 시작
#Input image size = (None,200,200,1)
input_shape=newmodel.input_shape

#Parallel 모델 합병
x1=layers.concatenate(results)

#Output layer addition
output_shape=layers.Dense(ran,activation="softmax",
                        kernel_initializer="glorot_normal",
                        kernel_regularizer=regularizers.l1_l2(l1=1e-2,l2=1e-2))(x1)

model=models.Model(inputs=input_shape,outputs=output_shape)

#Optimizer 설정
adam=optimizers.Adam(lr=1e-5)


#Model 실현화
model.compile(optimizer=adam,
            loss="categorical_hinge",
            metrics=["accuracy"])

#학습의 시작
hist=model.fit(x_train, y_train, batch_size=batch_sizes,
            epochs=epoch, verbose=1, shuffle="batch",
            validation_data=(x_val, y_val)) 

#정확도 출력
[loss,accuracy]=model.evaluate(x_test,y_test,verbose=1)
print("ACC : "+str(accuracy))

#모델 가중치 파일로 저장 (h5 type)
model.save("./Learning_Model_V9.h5",overwrite=True)
model.summary()

h5_f.close()

#정확도, Loss를 그래프화
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
