# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:28:13 2019

@author: Yue Xu
"""


from ToolBox import tools


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
from keras.utils import to_categorical
import hdf5storage

# Visualize training history
from keras import callbacks
from keras.callbacks import EarlyStopping

#tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32,
#                           write_graph=True, write_grads=True, write_images=False,
#                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# Early stopping  
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')


epochs = 200


t = hdf5storage.loadmat('X.mat')
t = t['X']
X =np.zeros((len(t[0,:]),len(t[0,0])))
for i in range(1,len(t[0,:])):
    X[i,:] = t[0,i].T

t = hdf5storage.loadmat('Y.mat')
Y = t['Y']
Y = to_categorical(Y)

len1 = np.shape(X)
testLen = int( len1[0]*0.1)

batch_size = testLen  

#x_test  = X[:testLen,:]
#y_test  = Y[:testLen,:]
x_test  = X[-2 * testLen:,:]
y_test  = Y[-2 * testLen:,:]
x_train = X[:-2 * testLen,:]
y_train = Y[:-2 * testLen,:]


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(5, activation='softmax'))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, \
          nesterov=True)
# ‘categorical_crossentropy’
model.compile(loss='mse',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle = True,
          validation_split=0.1, 
          )

model.summary()


score = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = np.sum(y_pred==y_test)*1.0/len(y_test)
print( "--- acc %s ---" %(acc))

import scipy.io as sio 
sio.savemat('DNN_Ypred.mat', {'y_pred':y_pred});
sio.savemat('DNN_Ytest.mat',{'y_test': y_test});