from keras.layers.regularization.spatial_dropout3d import Dropout
from keras.datasets import mnist
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential
import numpy as np
import random as rd 
import matplotlib.pyplot as plt
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
print(xtrain.shape)
xtrain1=xtrain.reshape(xtrain.shape[0],28,28,1)
xtest1=xtest.reshape(xtest.shape[0],28,28,1)
print(xtest1.shape)
model=Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2))) # it is return feature map 
model.add(Flatten())
model.add(Dense(28,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# train the model
model.fit(xtrain,ytrain,epochs=100)

# predict the model 
a=model.predict(xtest)

# testing and visualizing process 
index_r=10000
index=rd.randint(1,index_r)
b=a[index] # it is value of pic in th
j=np.argmax(b)
b=[0,0,0,0,0,0,0,0,0,0]
b[j]=1
print(b)
plt.imshow(xtest[index])

