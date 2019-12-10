import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Flatten, Lambda, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio
import random

np.random.seed(0)

# ======================== Parameters ========================

dataID = "./Data/m123-com/"
intType = 6  # 1-5, 6 for combine
print(dataID)
print(intType)
print( "===========================================")
if not os.path.isdir( dataID):
    os.mkdir( dataID)

epochs = 400   # number of learning epochs
batch_size = 128
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto') # Early stopping


#  =============== load data =================================
t = sio.loadmat(dataID+'X.mat')  # XLong1    XScale1
X = t['X']
t = sio.loadmat(dataID+'Y.mat')  #  YLong1   YScale1
Y= t['Y']


Nout = Y.shape[1]
print(Nout)
if intType == 1:
    X = X[:,[0,1,2]] # awgn
elif intType == 2:
    X = X[:,[0,1,2,6]] # CW
    X[:,3] = X[:,3]/(1000)
elif intType == 3:
    X = X[:,[0,1,2,7]] # Chirp
    X[:,3] = X[:,3]/(1000000)
elif intType == 4:
    X = X[:,[0,1,2,8,9,10]] # MOD
    X[:,3] = X[:,3]/(2)
    X[:,4] = X[:,4]/(100)
elif intType == 5:
    X = X[:,[0,1,2,11]] # FilNoise
    X[:,3] = X[:,3]/(1000)


# ================== Data processing ###################
# Ber in dB scale
sc_factor = 1
Y = Y/sc_factor
Y = - 10*(np.log(Y)/np.log(10));   # log value
for i in range(0,Y.shape[0]):
    for j in range(0,2):
        if np.isinf(Y[i,j]) or np.isnan(Y[i,j]) or Y[i,j] > 40 :
            Y[i,j] = 40

# split data
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

# extend dim
Xtrain = Xtrain[:, :, np.newaxis]
Xtest = Xtest[:, :, np.newaxis]
Xval = Xval[:, :, np.newaxis]

#================ Model Building ===========================
n_neurons = 128
nn_input  = Input((Xtrain.shape[1],1))
nn_output = Flatten()(nn_input)
nn_output = Dense(n_neurons,activation='relu')(nn_output)
# nn_output = BatchNormalization()(nn_output)
nn_output = Dense(n_neurons,activation='relu')(nn_output)
nn_output = Dense(n_neurons,activation='relu')(nn_output)
nn_output = Dense(n_neurons,activation='relu')(nn_output)
nn_output = Dense(Nout,activation='linear')(nn_output)  
nn = Model(inputs=nn_input,outputs=nn_output)


#================ Model Compiling ===========================
optz = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, amsgrad=False)
nn.compile(optimizer=optz, loss='mse',metrics=['mse'])
nn.summary()
train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                    batch_size = batch_size ,epochs = epochs ,\
                    validation_data=(Xval, Yval), shuffle=True, callbacks=[early_stop])

# ================ Evaluate Model  ===========================

Ypredtrain = nn.predict(Xtrain)
Ypred = nn.predict(Xtest)
sio.savemat(dataID+'Ytest.mat', {'Ytest':Ytest})
sio.savemat(dataID+'Ypred.mat', {'Ypred':Ypred})   

err = np.mean(abs(Ytest-Ypred),axis=0)
print( "--- MAE: --- %s" %(err))

YTEST = Ytest
YPRED = Ypred

for ID in range(0,Nout):
    Ytest = YTEST[:,ID]
    Ypred = YPRED[:,ID]
    cor = np.corrcoef(Ytest, Ypred)
    print(cor)
    print(dataID)
    print(ID)

    # scatter plot of Y_true VS Y_pred
    plt.figure(6)
    plt.scatter(Ytest, Ypred, facecolors='none', edgecolors='b')
    plt.title(' est vs ground')
    plt.ylabel('est')
    plt.xlabel('ground')
    plt.grid(True)
    plt.savefig(dataID+'scatter-o-%d.png'%ID)
    plt. clf()
    
    
    plt.figure(7)
    plt.scatter(Ytest, Ypred, s =1, facecolors='none', edgecolors='b')
    plt.title(' est vs ground')
    plt.ylabel('est')
    plt.xlabel('ground')
    plt.grid(True)
    plt.savefig(dataID+'scatter%d.png'%ID)
    plt. clf()
    
    
    
    plt.figure(2)
    plt.hist(Ytest-Ypred,bins=200)
    plt.ylabel('Number of occurence')
    plt.xlabel('Estimate error (deg)')
    plt.grid(True)  
    plt.title('histogram of estimation error')
    plt.savefig(dataID+'hist%d.png'%ID)
    plt. clf()
    
    with open(dataID+"result.txt", "a") as f:
        f.write('----- MAE: %s -------- \n'%err)
        f.write('----- ID: %s -------- \n'%ID)
        f.write('----- cor:  %s -------- \n'%cor)  
        f.write('\n') 
        f.write('\n') 
        f.write('\n') 
        print("check result.txt file")


#============== save model to file =============
# model_json = nn.to_json()
# with open("./Results/model_TF.json", "w") as json_file:
#     json_file.write(model_json)
#     nn.save_weights("./Results/model_TF.h5")
#     print("Saved model to disk")


