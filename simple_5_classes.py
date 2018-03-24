#简易5分类
#载入与模型网络构建
# WEIGHTS_PATH_NO_TOP 就是去掉了全连接层
from keras.models import Sequential
from  keras.layers import Dense,Activation,MaxPooling2D
from  keras.layers import Conv2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator#图片预处理
import matplotlib.pyplot as plt

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3),name='conv1_1'))
# filter大小3*3，数量32个，原始图像大小3,150,150
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),name='conv2_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),name='conv3_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())#this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))#输出层，几个分类就要有几个dense
model.add(Activation('softmax'))#多分类要用softmax,二分类不用
#二分类dense(2)+sigmoid(激活函数)
# input_shape=(3,150, 150)是theano的写法，而tensorflow需要写出：(150,150,3)；
# 需要修改Input_size。也就是”channels_last”和”channels_first”数据格式的问题。

#二分类
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#多分类:不是binary_crossentropy
#  优化器rmsprop：除学习率可调整外，建议保持优化器的其他默认参数不变
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#图像预处理，数据准备
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    'E:/keras_data/data1/train',
    target_size=(150,150),  # all images will be resized to 150x150
    batch_size=32,
    class_mode='categorical'#多分类
)

validation_generator=test_datagen.flow_from_directory(
    #计算数据的一些属性值，之后再训练阶段直接丢进去这些生成器
    'E:/keras_data/data1/test',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)
# 二分类class_mode='binary'
history_fit=model.fit_generator(
        train_generator,
        steps_per_epoch=100,#2000
        nb_epoch=3,#50
        validation_data=validation_generator,
        validation_steps=20)#800
# model.save_weights('E:/keras_data/data1/first_try_animal5.h5')
model.save('E:/keras_data/data1/5class_model.h5')
# samples_per_epoch,steps_per_epoch，相当于每个epoch数据量峰值，
# 每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束

#画图函数
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()
#训练的acc_loss图
plot_training(history_fit)
