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
import matplotlib.pyplot as plt
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
t = hdf5storage.loadmat('YLong1.mat')
Y = t['Y']

Y_unm = Y[:,0]/1000
Y_m = Y[:,1]/1000
Y_m = np.reshape(Y_m,(-1,1))
Y_m = tools.createClassfication(Y_m)
Y_m = np.reshape(Y_m,(-1,1))
plt.figure(1)
plt.hist(Y_m[:,0],bins=200)