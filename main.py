"""
#!/usr/bin/env python
# coding: utf-8
# Angle of Arrival Estimation 
# Jianyuan Jet Yu, jianyuan@vt.edu

# flexiable version w/ tensorflow
feature:
    1. early stop
    2. radius based function (RBF)
    3. drop out
    4. batch norm
    5. regularization


"""








import time
import os
import math




# ======================== global setting ======================== 

dataID = './Data/'    
note = ' TF digital, spread, 1 sig, SGD '  
Nsig = 1 



isSend       = 0     # send Email
epochs     = 4    # number of learning epochs
batch_size = 512
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"   # sometime make "1" for Mac 

TEXT    = "low dataset" 
num_bins = 50

tic = time.time()


import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=1):
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
import numpy as np
import hdf5storage
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv1D, Conv2D,\
        AveragePooling2D,Flatten, Dropout, SimpleRNN, LSTM, concatenate, Layer
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from IPython.core.display import Image, display
import matplotlib.pyplot as plt
from numpy import ma
import scipy.io as sio
from IPython.display import Image
from matplotlib import cm as CM
from nbconvert import HTMLExporter
import keras
keras.__version__

# Visualize training history
from keras import callbacks
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=False,
                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# Early stopping  
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

from datetime import datetime
import scipy.io as sio

from keras.models import model_from_json

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


#!nvidia-smi

# Distance Functions
def true_dist(y_true, y_pred):
    return np.sqrt(np.square(np.abs(y_pred[:,0]-y_true[:,0]))+ np.square(np.abs(y_pred[:,1]-y_true[:,1])) )
    # to be amend

def dist(y_true, y_pred):    
     return tf.reduce_mean((tf.sqrt(tf.square(tf.abs(y_pred[:,0]-y_true[:,0]))+ tf.square(tf.abs(y_pred[:,1]-y_true[:,1])))))  

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


#  =============== 

# Example for Measurement Quality

  
    
t = hdf5storage.loadmat(dataID+'Xtrain.mat')
Xtrain = t['Xtrain'].T  # Nsnr X Ndat X 42
t = hdf5storage.loadmat(dataID+'Ytrain.mat')
Ytrain = t['Ytrain'].T  # Nsnr X Ndat X dim X 2
t = hdf5storage.loadmat(dataID+'Xtest.mat')
Xtest = t['Xtest'].T 
t = hdf5storage.loadmat(dataID+'Ytest.mat')
Ytest = t['Ytest'].T


with open("outPut_TF.txt", "a") as text_file:

    text_file.write( "=== " + note + " === \n" )
    text_file.write( "--- caseID %s  begin --- \n" %(dataID))
    text_file.write( "--- local time  " + dt_string + " --- \n" )
    Ytest = Ytest

    Xval = Xtrain[0:20,:];
    Yval = Ytrain[0:20];
    Xtrain = Xtrain[21:,:];
    Ytrain = Ytrain[21:];
    
    # extend dim
    Xtrain = Xtrain[:,:,np.newaxis]
    Xtest = Xtest[:,:,np.newaxis]
    Xval = Xval[:,:,np.newaxis]

    nn_input  = Input((30,1))
                
    nn_output = Flatten()(nn_input)
    # nn_output = LSTM(units=136, dropout_U = 0.2, dropout_W = 0.2)(nn_input)
    #nn_output = Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.001))(nn_output)
#        nn_output = RBFLayer(64, 0.5)(nn_output)
    nn_output = Dense(128,activation='relu')(nn_output)  
#        nn_output = Dropout(0.1)(nn_output)
    nn_output = Dense(256,activation='relu')(nn_output)  
#        nn_output = Dropout(0.1)(nn_output)
#        nn_output = Dropout(0.2)(nn_output)
#        nn_output = Dense(2048,activation='relu')(nn_output) 
#        nn_output = Dropout(0.2)(nn_output)
#        nn_output = Dense(4096,activation='relu')(nn_output)
    nn_output = Dense(32,activation='relu')(nn_output)
#        nn_output = Dropout(0.1)(nn_output)
#        nn_output = Dropout(0.2)(nn_output)
    nn_output = Dense(2*Nsig,activation='linear')(nn_output)
    #  directly output 3 para, non classify
    nn = Model(inputs=nn_input,outputs=nn_output)
    
    nn.compile(optimizer='adam', loss='mse',metrics=[dist])
    nn.summary()

    train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                        batch_size = batch_size ,epochs = epochs ,\
                        validation_data=(Xval, Yval), \
                        shuffle=True,\
                        callbacks=[early_stop])
            
    # Evaluate Performance
    Ypredtrain = nn.predict( Xtrain)
    Ypred = nn.predict( Xtest)
    
    Ypredtrain = Ypredtrain/math.pi*180
    Ypred = Ypred/math.pi*180
    Ytest = Ytest/math.pi*180
    Ytrain = Ytrain/math.pi*180
    
    
    err = np.mean(abs(Ytest-Ypred),axis=0);
    print( "--- MAE: --- %s" %(err))
    
    text_file.write( " layer [512] \n")
    text_file.write( " test error %s (deg) \n" %(err))


    
#        errors_train = true_dist(Ytrain,Ypredtrain)
#        errors_test  = true_dist(Ytest,Ypred)
       
#        Mean_Error_Train =  np.mean(np.abs(errors_train))
#        Mean_Error_Test= np.mean(np.abs(errors_test),0)
#        print("Mean error on Train area:", Mean_Error_Train)
#        print("Mean error on Test  area:",Mean_Error_Test)
    
    
    toc =  time.time()
    timeCost = toc - tic
    print( "--- Totally %s seconds ---" %(timeCost))
    text_file.write( " timeCost %s \n" %(timeCost))
    
    # write log to file
#        text_file.write( "SNR index %d, time cost %s, test error %s (deg) \n" %(i,timeCost, Mean_Error_Test))

    # save model to file
    model_json = nn.to_json()
    with open(dataID+"model_TF.json", "w") as json_file:
        json_file.write(model_json)
        nn.save_weights(dataID+"model_TF.h5")
        print("Saved model to disk")
    
    
    
    # Histogramm of errors on test Area
    plt.figure(2)
   
    plt.hist(err,bins=64)
    plt.ylabel('Number of occurence')
    plt.xlabel('Estimate error (deg)')
    plt.grid(True)  
    plt.title('histogram of estimation error')
    plt.savefig(dataID+'hist_TF.png')
    
#        num_bins = 50
#        plt.figure(3)
#        counts, bin_edges = np.histogram(errors, bins=num_bins)
#        cdf = np.cumsum(counts)/np.sum(counts)
#        plt.plot(bin_edges[1:], cdf)
#        plt.xlabel('est error (deg)')
#        plt.ylabel('F(X<x)')
#        plt.grid(True)  
#        plt.title('Cdfplot of distance error')
#        plt.savefig('./Figpy/err_cdf.png')
    
    
    #  =============== 
    # Error Vector over Area in XY
    #error_vectors = np.real(Ypred - Ytest)
    #plt.figure(4)
    #plt.quiver(np.real(Ytest[:,0]),np.real(Ytest[:,1]),error_vectors[:,0],error_vectors[:,1],errors)
    #plt.xlabel("x in m")
    #plt.ylabel("y in m")
    #plt.grid(True) 
    #plt.title('quiver of distance error') 
    #plt.savefig('position.png')
    
    
#        plt.figure(5)
#        plt.plot(train_hist.history['dist'])
#        plt.plot(train_hist.history['val_dist'])
#        plt.title('distance')
#        plt.ylabel('distance')
#        plt.xlabel('epoch')
#        plt.grid(True)  
#        plt.legend(['train', 'validate'])
#        plt.savefig('./Figpy/hist_dist.png')
    
#        plt.figure(6)
#        plt.plot(train_hist.history['loss'])
#        plt.plot(train_hist.history['val_loss'])
#        plt.title('model loss')
#        plt.ylabel('loss')
#        plt.xlabel('epoch')
#        plt.grid(True)  
#        plt.legend(['train', 'validate'])
#        plt.savefig('./Figpy/hist_loss.png')
    
    
#        dim = 0
#        plt.figure(7)
#        plt.scatter( Ytrain[:,dim], Ypredtrain[:,dim],facecolors='none',edgecolors='b')
#        plt.title('dim%d train - est vs ground'%dim)
#        plt.ylabel('est')
#        plt.xlabel('ground')
#        plt.grid(True)  
#        plt.savefig('./Figpy/dim%d_train.png'%dim)
    
    plt.figure(8)
    plt.scatter( Ytest[:,0], Ypred[:,0],facecolors='none',edgecolors='b')
    plt.scatter( Ytest[:,1], Ypred[:,1],facecolors='none',edgecolors='r')
    plt.title('Nsig%d test - est vs ground'%Nsig*2)
    plt.ylabel('est')
    plt.xlabel('ground')
    plt.grid(True)  
    plt.savefig(dataID+'scatter_TF.png')
        
#        dim = 1
#        plt.figure(9)
#        plt.scatter( Ytrain[:,dim], Ypredtrain[:,dim],facecolors='none',edgecolors='b')
#        plt.title('dim%d train - est vs ground'%dim)
#        plt.ylabel('est')
#        plt.xlabel('ground')
#        plt.grid(True)  
#        plt.savefig('./Figpy/dim%d_train.png'%dim)
#        
#        plt.figure(10)
#        plt.scatter( Ytest[:,dim], Ypred[:,dim],facecolors='none',edgecolors='b')
#        plt.title('dim%d test - est vs ground'%dim)
#        plt.ylabel('est')
#        plt.xlabel('ground')
#        plt.grid(True)  
#        plt.savefig('./Figpy/dim%d_test.png'%dim)
    
    text_file.write("--- caseID %s  end --- \n" %(dataID))
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")
    
# save down pred data
sio.savemat(dataID+'Ypred_TF.mat', {'Ypred':Ypred})


import yagmail
FROM = 'yujianyuanhaha@gmail.com'
TO = 'yujianyuanhaha@gmail.com'

SUBJECT = '[adam+ MAE]AoA, TF.py, testErr %s'%(err)

ATTACHMENTS = [
           dataID+'hist_TF.png',dataID+'scatter_TF.png',\
           'mainTF.py','outPut_TF.txt',dataID+'Ypred_TF.mat'
           ]

yag = yagmail.SMTP(FROM, '!Yu3843526')
if isSend:
    yag.send(TO, SUBJECT, TEXT,ATTACHMENTS)