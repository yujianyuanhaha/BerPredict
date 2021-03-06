# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:28:13 2019

@author: Yue Xu
"""
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from ToolBox import tools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.utils import to_categorical
import hdf5storage
from keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ========================= Parameter Settings ================

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')  # Early stopping  for validation
flag_ber = 1 #0 is unmit data; 1 is mit data
epochs = 100
batch_size = 512
dataID = './Data/'
# ========================= Loading and processing Data ================

t = hdf5storage.loadmat('X.mat')
X = t['X']

#add_type_len = len(X[:,1])  # adding interference mitigation type to our features [0,1,2,3,4,5]
#add_type = np.zeros((add_type_len,1))
#for i in range(add_type_len):
#    add_type[i,:] = 1
#X = np.array(X)
#X_full = np.column_stack((X,add_type))
#X_temp = np.column_stack((X[:,0],X[:,1]))
#X_temp = np.column_stack((X_temp,X[:,2]))
#X_temp = np.column_stack((X_temp,X[:,3]))
#X_temp = np.column_stack((X_temp,X[:,5]))
#X_temp = np.column_stack((X_temp,X[:,6]))
#X_temp = np.column_stack((X_temp,X[:,7]))
#X_temp = np.column_stack((X_temp,X[:,8]))
#X_temp = np.column_stack((X_temp,X[:,9]))

#X_full = X 
#X_full = X[:,[0,1,2,4,5]]
#x_full = X[:,[0,1,2,4,5]]

#X_full = X[:,[0,1,2,4,5]] #AWGN

#X_full = X[:,[0,1,2,4,5,6]] #CW
#X_full[:,5]/=1000

#X_full = X[:,[0,1,2,4,5,7]]  #CHIRP
#X_full[:,5]/=100000

#X_full = X[:,[0,1,2,4,5,8,9,10]]  #MOD
#X_full[:,6]/=100
#X_full = X[:,[0,1,2]]  #Filter



#n_order = np.array([1,2,3]) # normalized feature
#for j in range(len(n_order)):
#    temp = X[:,n_order[j]]
#    temp = np.array(temp)
#    maxV = max(temp)
#    minV = min(temp)
#    X[:,n_order[j]] = (X[:,n_order[j]] - minV) / (maxV - minV)
#
#
#
#
#for i in range(len(X[:,1])):
#    if X[i,4] == 1:
#        X[i,[3,5,6,7,8,9,10]] = 0
#    elif X[i,4] == 2:
#        X[i,[3,6,7,8,9,10]] = 0
#    elif X[i,4] == 3:
#        X[i,[3,5,7,8,9,10]] = 0
#    elif X[i,4] == 4:
#        X[i,[3,5,6,10]] = 0
#    else:
#        X[i,[3,5,6,7,8,9]] = 0
#        
#X[:,8]/=100

X_full = X     
           


t = hdf5storage.loadmat('Y.mat')
Y = t['Y']

# Y_unm = Y[:,0]/1000  # normalize unmitigated BER (it was generated in matlab with a scale of 1000)
Y_m = Y[:,flag_ber] # normalize mitigated BER (it was generated in matlab with a scale of 1000)
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


temp_in = x_train.shape[1]
temp_out = Y_m.shape[1]

# ========================= Loading and processing Data ================

model = Sequential()
model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(Y_m.shape[1], activation='softmax'))

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
avgqerr = tools.calculateQBER2(y_qber_est,y_qber_real)
print( "--- acc %s ---" %(acc))
print( "--- avgqerr %s ---" %(avgqerr))

corr = np.corrcoef(y_pred, y_test)
print("--- corr(level) %s ---" %(corr))
y_qber_est2 = np.reshape(np.array(y_qber_est),(1,testLen))
y_qber_real2 = np.reshape(np.array(y_qber_real),(1,testLen))
corr2 = np.corrcoef(y_qber_est2, y_qber_real2)
print("--- corr(realValue) %s ---" %(corr2))


plt.figure(1)
plt.hist(y_qber_est,bins=200)
plt.xlabel('Estinated Ber Level (q)')
plt.ylabel('Quantity')
plt.savefig('./Results/histogram_estimated_BER.png')
plt.figure(2)
plt.hist(y_qber_real,bins=200)
plt.xlabel('Real Ber Level (q)')
plt.ylabel('Quantity')
plt.savefig('./Results/histogram_real_BER.png')
plt.figure(3)
plt.hist(y_test,bins=200)
plt.savefig('./Results/histogram_estimated_BER.png')



cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(4)
plt.imshow(cm, cmap='binary')
plt.savefig('./Results/Confusion_matrix.png')

tools.cm_analysis(y_test, y_pred, ymap=None, figsize=(10,10))
