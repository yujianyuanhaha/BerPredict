#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:21:58 2019

@author: Jet
"""

import numpy as np
import hdf5storage
from tensorflow.keras.models import model_from_json
import scipy.io as sio
from sklearn.externals import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ---------------- load data ---------------------------------
t = hdf5storage.loadmat('./DATA/X160k.mat')
X_test = t['X']

# ---------------- load model ---------------------------------
json_file = open('./Results/model_TF.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_final = model_from_json(loaded_model_json)
model_final.load_weights('./Results/model_TF.h5')

# ---------------- offline test ---------------------------------
X_test = X_test[:, :, np.newaxis]
Ypred = model_final.predict(X_test)

sio.savemat('./predict/Ypred_TF.mat', {'y_pred':Ypred})
print('done')