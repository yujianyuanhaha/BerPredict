#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:09:17 2019

@author: Jet


1. TODO, lack of early stop, likely overfit

"""



import time
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"   # sometime make "1" for Mac 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import to_categorical
import hdf5storage
from keras import callbacks
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
 
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# ---------------- settings---------------------------------
epochs = 20
ID = 0    # 0 for umit, 1 for mit
MAXdB = 40
batch_size = 4096


# ---------------- load data ---------------------------------
t      = hdf5storage.loadmat('./DATA/X_200k.mat')
X  = t['X'] 
t      = hdf5storage.loadmat('./DATA/Y_200k.mat')
U  = t['Y']
Ndat = U.shape[0]

U = - 10*(np.log(U/1000)/np.log(10));   # log value
for i in range(0,Ndat):
    for j in range(0,2):
        if np.isinf(U[i,j]) or np.isnan(U[i,j]) or U[i,j] > MAXdB:
            U[i,j] = MAXdB
        U[i,j] = int(U[i,j]/2.50)

plt.figure(2)
plt.hist(U[:,1],bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('BER (-10log)')
plt.grid(True)  
plt.title('histogram of BER')
plt.savefig('./hist_BER_umit_float.png')

Y = to_categorical(U[:,ID] )
numClass = Y.shape[1]

Xtest = X[ int(X.shape[0]*0.8):,:]
Ytest = Y[ int(X.shape[0]*0.8):]
Xtrain = X[ :int(X.shape[0]*0.8),:]
Ytrain = Y[ :int(X.shape[0]*0.8)]


# ---------------- constrcut NN ---------------------------------
tic = time.time()

model = Sequential()
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(numClass, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, \
          nesterov=True)

model.compile(loss='mse',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain,
          epochs=epochs,
          batch_size=batch_size,
          shuffle = True,
          validation_split=0.1, 
          callbacks=[early_stop])

model.summary()

toc =  time.time()
timeCost = toc - tic
print( "--- Totally %s seconds ---" %(timeCost))

# ---------------- eval NN ---------------------------------

score = model.evaluate(Xtest, Xtest, batch_size=128)
y_pred = model.predict(Xtest)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(Ytest,axis=1)
acc = np.sum(y_pred==y_test)*1.0/len(y_test)
print( "--- acc %s ---" %(acc))

import scipy.io as sio 
sio.savemat('DNN_Ypred.mat', {'y_pred':y_pred});
sio.savemat('DNN_Ytest.mat',{'y_test': y_test});
