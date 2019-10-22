"""
#!/usr/bin/env python
# coding: utf-8
# BER predict
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import hdf5storage # load mat files
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Flatten, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio

np.random.seed(0)
#tf.random.set_seed(0)

# import keras
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# print(tf.__version__)
# print(tf.keras.__version__)
# from tensorflow.keras import backend as K

#
# def get_session(gpu_fraction=1):
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#
# KTF.set_session(get_session())


# ======================== global setting ========================

dataID = './Data/'
Nsig = 1
epochs = 100   # number of learning epochs
batch_size = 512
num_bins = 50
note = ''


tic = time.time()
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

#  =============== load data =================================

t = hdf5storage.loadmat(dataID+'X_200k.mat')  # XLong1    XScale1
X = t['X']
t = hdf5storage.loadmat(dataID+'Y_200k.mat')  #  YLong1   YScale1
Y= t['Y']

# ================== Data processing ###################

# either scale and transfer to dB

sc_factor = 1
Y = Y/sc_factor

# eps = 10**-12
# Y = -10*(np.log(Y/1000 + eps)/np.log(10))   # log value

#Y = np.sort(Y,axis=1)


Y = - 10*(np.log(Y/1000)/np.log(10));   # log value
for i in range(0,Y.shape[0]):
    for j in range(0,2):
        if np.isinf(Y[i,j]) or np.isnan(Y[i,j]):
            Y[i,j] = 100







train_fraction = 0.7
train_size = int(train_fraction*Y.shape[0])
Xtrain = X[:train_size,:]
Ytrain = Y[:train_size,:]


val_fraction = 0.5
val_size = int((Y.shape[0] - train_size)*val_fraction)
Xval = X[train_size:train_size+val_size, :]
Yval = Y[train_size:train_size+val_size,:]

Xtest = X[train_size+val_size:,:]
Ytest = Y[train_size+val_size:,:]



#
# plt.figure(1)
# plt.hist(Ytest[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_BER_test_mit.png')
#
#
#
#
# plt.figure(2)
# plt.hist(Ytest[:,1],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test umit.')
# plt.savefig('./Results/hist_BER_test_umit.png')
#
#
#
#
# plt.figure(1)
# plt.hist(Yval[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_BER_val_mit.png')
#
#
#
#
# plt.figure(2)
# plt.hist(Yval[:,1],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test umit.')
# plt.savefig('./Results/hist_BER_val_umit.png')
#
#
#
# plt.figure(3)
# plt.hist(Xval[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_BER_val_mit1.png')
#
#
#
#
# plt.figure(3)
# plt.hist(Xtest[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y_test')
# plt.grid(True)
# plt.title('histogram of BER_test umit.')
# plt.savefig('./Results/hist_BER_val_umit1.png')


# Early stopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# # Distance Functions
# def dist(y_true, y_pred):
#      return tf.reduce_mean(tf.sqrt(tf.square(tf.abs(y_pred-y_true)) ))

def clip(x,fact=sc_factor):
    return tf.keras.backend.clip(x,0.0,1000.0/fact)



with open("./Results/outPut_TF.txt", "a") as text_file:

    text_file.write( "=== " + note + " === \n" )
    text_file.write( "--- caseID %s  begin --- \n" %(dataID))
    text_file.write( "--- local time  " + dt_string + " --- \n" )


    # extend dim
    Xtrain = Xtrain[:,:,np.newaxis]
    Xtest = Xtest[:,:,np.newaxis]
    Xval = Xval[:,:,np.newaxis]


    nn_input  = Input((9,1))
    nn_output = Flatten()(nn_input)
    nn_output = Dense(32,activation='relu')(nn_output)
    nn_output = Dense(32,activation='relu')(nn_output)
    nn_output = Dense(32,activation='relu')(nn_output)
    nn_output = Dense(2,activation='linear')(nn_output)
    # nn_output = Dense(2, activation='linear')(nn_output)
    # nn_output = Lambda(clip)(nn_output)

    nn = Model(inputs=nn_input,outputs=nn_output)

    # nn.compile(optimizer='adam', loss='mse',metrics=[dist])
    optz = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, amsgrad=False)
    nn.compile(optimizer=optz, loss='mse',metrics=['mse'])
    # nn.compile(optimizer='adam', loss='mse', metrics=['mse'])

    nn.summary()


    train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                        batch_size = batch_size ,epochs = epochs ,\
                        validation_data=(Xval, Yval), shuffle=True, callbacks=[early_stop])

    # Evaluate Performance
    Ypredtrain = nn.predict(Xtrain)
    Ypred = nn.predict(Xtest)


    err = np.mean(abs(Ytest-Ypred),axis=0)
    print( "--- MAE: --- %s" %(err))
    print(np.corrcoef(Ytest[:,0], Ypred[:,0]))
    print(np.corrcoef(Ytest[:,1], Ypred[:,1]))

    # text_file.write( " layer [512] \n")
    # text_file.write( " test error %s (deg) \n" %(err))

    toc =  time.time()
    timeCost = toc - tic
    print( "--- Totally %s seconds ---" %(timeCost))
    text_file.write( " timeCost %s \n" %(timeCost))


    # save model to file
    model_json = nn.to_json()
    with open("./Results/model_TF.json", "w") as json_file:
        json_file.write(model_json)
        nn.save_weights("./Results/model_TF.h5")
        print("Saved model to disk")



    # Histogramm of errors on test Area
    plt.figure(2)
    plt.hist(abs(Ytest[:,0]-Ypred[:,0]),bins=64)
    plt.ylabel('Number of occurence')
    plt.xlabel('Estimate error')
    plt.grid(True)
    plt.title('histogram of estimation error mit.')
    plt.savefig('./Results/hist_error_mit.png')

    plt.figure(3)
    plt.hist(abs(Ytest[:, 1] - Ypred[:, 1]), bins=64)
    plt.ylabel('Number of occurence')
    plt.xlabel('Estimate error')
    plt.grid(True)
    plt.title('histogram of estimation error unmit.')
    plt.savefig('./Results/hist_error_unmit.png')

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


    plt.figure(5)
    #plt.plot(train_hist.history['mse'])
#
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
    plt.scatter( Ytest[:100,0], Ypred[:100,0],facecolors='none',edgecolors='b')
    plt.scatter( Ytest[:100,1], Ypred[:100,1],facecolors='none',edgecolors='r')
    plt.title(' est vs ground')
    plt.ylabel('est')
    plt.xlabel('ground')
    plt.legend(['mitBER','umitBER'])
    plt.grid(True)  
    plt.savefig('./Results/scatter_TF.png')
        
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
