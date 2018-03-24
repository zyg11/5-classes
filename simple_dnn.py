#案例一：简单的单层-全连接网络
from keras.models import Model
from keras.layers import Dense,Input
inputs=Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x=Dense(64,activation='relu')(inputs)
# 输入inputs，输出x
# (inputs)代表输入
predictions=Dense(10,activation='softmax')(x)
# 输入x，输出分类

# This creates a model that includes
# the Input layer and three Dense layers
model=Model(inputs=inputs,outputs=predictions)#该句是函数式模型的经典，
# 可以同时输入两个input，然后输出output两个模型
model.compile(optimizer='remsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data,label)# starts training