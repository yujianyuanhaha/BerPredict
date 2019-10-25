# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:28:13 2019

@author: Yue Xu
"""
from sklearn.metrics import confusion_matrix
from ToolBox import tools
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np
from tensorflow.keras.utils import to_categorical
import hdf5storage
from tensorflow.keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ========================= Parameter Settings ================

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')  # Early stopping  for validation
epochs = 50
batch_size = 512
dataID = './Data/'
# ========================= Loading and processing Data ================

t = hdf5storage.loadmat(dataID+'X160k_shuffled.mat')
X = t['X']

add_type_len = len(X[:,0])  # adding interference mitigation type to our features [0,1,2,3,4,5]
add_type = np.zeros((add_type_len,1))
for i in range(add_type_len):
    add_type[i,:] = 1
X = np.array(X)
X_full = np.column_stack((X,add_type))

t = hdf5storage.loadmat(dataID+'Y160k_shuffled.mat')
Y = t['Y']

# Y_unm = Y[:,0]/1000  # normalize unmitigated BER (it was generated in matlab with a scale of 1000)
Y_m = Y[:,1]/1000  # normalize mitigated BER (it was generated in matlab with a scale of 1000)
Y_m = np.reshape(Y_m,(-1,1))
y_real = Y_m # save the real BER value for later evaluation
Y_m = tools.createClassfication(Y_m) # quantize the BER
Y_m = np.reshape(Y_m,(-1,1))
Y_m = to_categorical(Y_m)  # convert the levels into class labels  [0-18]

# splitting into training and testing
len1 = np.shape(X_full)
testLen = int( len1[0]*0.2)
x_test  = X_full[-testLen:,:]
y_test  = Y_m[-testLen:,:]
x_train = X_full[:-testLen,:]
y_train = Y_m[:-testLen,:]
y_qber_real = y_real[-testLen:,:] # true BER to be used in testing (calculating quantization error)



# ========================= Loading and processing Data ================

model = Sequential()
model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(19, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) # 'mse'
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle = True)

model.summary()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
y_qber_est = tools.mapBERvalue(y_pred)
acc = np.sum(y_pred==y_test)*1.0/len(y_test)
avgqerr = tools.calculateQBER(y_qber_est,y_qber_real)
print( "--- acc %s ---" %(acc))
print( "--- avgqerr %s ---" %(avgqerr))


plt.figure(1)
plt.hist(y_qber_est,bins=200)
plt.savefig('./Results/histogram_estimated_BER.png')
plt.figure(2)
plt.hist(y_qber_real,bins=200)
plt.savefig('./Results/histogram_real_BER.png')
# plt.figure(3)
# plt.hist(y_test,bins=200)
# plt.savefig('./Results/histogram_estimated_BER.png')


cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(4)
plt.imshow(cm, cmap='binary')
plt.savefig('./Results/Confusion_matrix.png')

tools.cm_analysis(y_test, y_pred, ymap=None, figsize=(10,10))
