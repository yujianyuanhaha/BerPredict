# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 07:42:16 2019

@author: Yue Xu
"""






import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
from keras.utils import to_categorical
import hdf5storage


epochs = 40

t = hdf5storage.loadmat('X.mat')
t = t['X']
X =np.zeros((len(t[0,:]),len(t[0,0])))
for i in range(1,len(t[0,:])):
    X[i,:] = t[0,i].T

t = hdf5storage.loadmat('Y.mat')
Y = t['Y']
Y = to_categorical(Y)

len1 = np.shape(X)
testLen = int( len1[0]*0.10)


batch_size = testLen  

x_test  = X[:testLen,:]
y_test  = Y[:testLen,:]
x_train = X[testLen+1:,:]
y_train = Y[testLen+1:,:]


x_test = x_test[:,:,np.newaxis]
#y_test = y_test[:,:,np.newaxis]
x_train = x_train[:,:,np.newaxis]
#y_train = y_train[:,:,np.newaxis]


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))



model.add(LSTM(256, return_sequences=True,
               input_shape=(500, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(256))  # return a single vector of dimension 32


model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, \
          nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size)

print( "--- Totally %s seconds ---" %(timeCost))
score = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = np.sum(y_pred==y_test)*1.0/len(y_test)
print( "--- acc %s ---" %(acc))