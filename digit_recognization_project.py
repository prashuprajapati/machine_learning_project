from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random

# split the our data 
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

# Dense used for creatin any layer
# Flatten used for convert from 2D into 1D array 
model=Sequential()
model.add(Flatten(input_shape=(28,28))) # size of image (28,28)
model.add(Dense(28,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10)

# predict the our model 
model.predict(xtest)
print(len(xtrain))

#train the model and show the digit 
index_r=10000
index=random.randint(1,index_r)
print('index is : ',index)
a=model.predict(xtest[index])
j=np.argmax(a)
a=[0,0,0,0,0,0,0,0,0,0]
a[j]=1
plt.imshow(xtest[index])
print(a)
