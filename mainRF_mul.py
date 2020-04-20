#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:02:14 2020

@author: jianyuan
"""

import time
import os
import numpy as np
import hdf5storage # load mat files
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio
import random
from sklearn.ensemble import RandomForestRegressor

np.set_printoptions(precision=2)

np.random.seed(0)
# tf.random.set_seed(0)
import time

start = time.time()

# ======================== Parameters ========================

# RF paramas
max_depth = 20
random_state = 1
dataID = './Data/all2/'
method = 'RF_100_mul/'
figName = ['unMit','FFT2','D3S','Notch','FRFT'];

intType = 6  # 1-5, 6 for combine
print(dataID)
print(method)
print(intType)

print( "===========================================")

if not os.path.isdir( dataID+method):
    os.mkdir( dataID+method)


#  =============== load data =================================
t = hdf5storage.loadmat(dataID+'X.mat')  # XLong1    XScale1
X = t['X']
t = hdf5storage.loadmat(dataID+'Y.mat')  #  YLong1   YScale1
Y= t['Y']

# duty of, SNR off, mat 34 -> py 23
# X = X[:,[0,1,4,5,6,7,8,9,10,11,12,13]]   # last one isD3S


# ================== Data processing ###################
# Ber in dB scale
sc_factor = 1
Y = Y/sc_factor

Y = - 10*(np.log(Y)/np.log(10));   # log value
for i in range(0,Y.shape[0]):
    for j in range(0,2):
        if np.isinf(Y[i,j]) or np.isnan(Y[i,j]) or Y[i,j] > 50 :
            Y[i,j] = 50

# split data
train_fraction = 0.7
train_size = int(train_fraction*Y.shape[0])
X_train = X[:train_size,:]
Y_train = Y[:train_size,:]
val_fraction = 0.5
val_size = int((Y.shape[0] - train_size)*val_fraction)
X_val = X[train_size:train_size+val_size, :]
Y_val = Y[train_size:train_size+val_size,:]
X_test = X[train_size+val_size:,:]
Y_test = Y[train_size+val_size:,:]
sio.savemat(dataID+'X_test.mat', {'X_test':X_test})
sio.savemat(dataID+'Y_test.mat', {'Y_test':Y_test})






#print( "--- ID: --- %d" %(ID))
Ytrain = Y_train
Ytest = Y_test
Yval = Y_val
Xtrain = X_train
Xtest = X_test
Xval = X_val


regr = RandomForestRegressor(n_estimators=100,max_depth=max_depth, random_state=random_state)
regr.fit(X_train, Ytrain)
print(regr.feature_importances_)



# ================ Evaluate Model  ===========================
Ypred = regr.predict(Xtest)
sio.savemat(dataID+method+'Ytest.mat', {'Ytest':Ytest})
sio.savemat(dataID+method+'Ypred.mat', {'Ypred':Ypred})    # <<<<<<<<<
err = np.mean(abs(Ytest-Ypred),axis=0)
print( "--- MAE: --- %s" %(err))
cor = np.corrcoef(Ytest, Ypred)
print(cor)
print(dataID)



for ID in range(0,Y.shape[1]): #########################

   

    # scatter plot of Y_true VS Y_pred
    plt.figure(6)
    plt.scatter(Ytest[:,ID], Ypred[:,ID], facecolors='none', edgecolors='b')
    # plt.scatter(Ytest[:100, 1], Ypred[:100, 1], facecolors='none', edgecolors='r')
    plt.title( figName[ID]+'\n'+'  MAE %.2f,'%err[ID] )
    plt.ylabel('predict Ber (dB)')
    plt.xlabel('ground Ber(dB)')
    plt.grid(True)
    plt.savefig(dataID+method+'scatter-o-%d.png'%ID)
    plt. clf()
    
    
    plt.figure(7)
    plt.scatter(Ytest[:,ID], Ypred[:,ID], s =1, facecolors='none', edgecolors='b')
    # plt.scatter(Ytest[:100, 1], Ypred[:100, 1], facecolors='none', edgecolors='r')
    plt.title( figName[ID]+'\n'+'  MAE %.2f,'%err[ID] )
    plt.ylabel('predict Ber (dB)')
    plt.xlabel('ground Ber(dB)')
    plt.grid(True)
    plt.savefig(dataID+method+'scatter%d.png'%ID)
    plt. clf()

#    plt. clf()
    
    with open(dataID+method+"result.txt", "a") as f:
        
        f.write('----- ID: %s -------- \n'%ID)
        f.write('--- randForest importace %s---'%regr.feature_importances_)
        f.write('\n') 
        f.write('\n') 
        f.write('\n') 
        
        print("check result.txt file")
    
end = time.time()   
print('time %s'%(end-start))

