#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:21:58 2019

@author: Jet
"""

import numpy as np
import xgboost as xgb
from sklearn.externals import joblib
import hdf5storage
import matplotlib.pyplot as plt
import math
from keras.models import model_from_json
import scipy.io as sio

# ---------------- load data ---------------------------------
t      = hdf5storage.loadmat('./DATA/X_200k.mat')
X  = t['X'] 
t      = hdf5storage.loadmat('./DATA/Y_200k.mat')
Y  = t['Y']

Y = - 10*(np.log(Y/1000)/np.log(10));   # log value
for i in range(0,200000):
    for j in range(0,2):
        if np.isinf(Y[i,j]):
            Y[i,j] = 100

Xtest = X[ int(X.shape[0]*0.8):,:]
Ytest = Y[ int(X.shape[0]*0.8):,:]


# ---------------- load model ---------------------------------

json_file = open('./DATA/model_TF.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
tf = model_from_json(loaded_model_json)
tf.load_weights('./DATA/model_TF.h5')

# ---------------- offline test ---------------------------------
Xtest = Xtest[:,:,np.newaxis]
Ypred = tf.predict(Xtest)

sio.savemat('./DATA/Ypred_TF.mat', {'y_pred':Ypred})


# ---------------- evaluation ---------------------------------
plt.figure(8)
plt.scatter( Ytest[:,0], Ypred[:,0],facecolors='none',edgecolors='b')
plt.scatter( Ytest[:,1], Ypred[:,1],facecolors='none',edgecolors='r')
plt.title(' est vs ground')
plt.ylabel('est')
plt.xlabel('ground')
plt.legend(['umitBER','mitBER'])
plt.grid(True)  
plt.savefig('./scatter_TF.png')