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
tf.random.set_seed(0)

# ======================== Parameters ========================

dataID = './Data/'
epochs = 100   # number of learning epochs
batch_size = 512
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto') # Early stopping

#  =============== load data =================================

t = hdf5storage.loadmat(dataID+'X_200k.mat')  # XLong1    XScale1
X = t['X']
t = hdf5storage.loadmat(dataID+'Y_200k.mat')  #  YLong1   YScale1
Y= t['Y']

# ================== Data processing ###################

# back to BER from 0 to 1

sc_factor = 1000
Y = Y/sc_factor

#Y = np.sort(Y,axis=1)

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


# make sure data has same distribution

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


# extend dim
Xtrain = Xtrain[:, :, np.newaxis]
Xtest = Xtest[:, :, np.newaxis]
Xval = Xval[:, :, np.newaxis]

#================ Extra Functions ===========================

# # Distance Functions
# def dist(y_true, y_pred):
#      return tf.reduce_mean(tf.sqrt(tf.square(tf.abs(y_pred-y_true)) ))

# def clip(x,fact=sc_factor):
#     return tf.keras.backend.clip(x,0.0,1000.0/fact)


#================ Model Building ===========================



nn_input  = Input((9,1))
nn_output = Flatten()(nn_input)
nn_output = Dense(1024,activation='tanh')(nn_output)
nn_output = Dense(1024,activation='tanh')(nn_output)
nn_output = Dense(1024,activation='tanh')(nn_output)
nn_output = Dense(2,activation='sigmoid')(nn_output)
# nn_output = Dense(2, activation='linear')(nn_output)
# nn_output = Lambda(clip)(nn_output)

nn = Model(inputs=nn_input,outputs=nn_output)


#================ Model Compiling ===========================


optz = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, amsgrad=False)
# optz = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
# optz = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)
nn.compile(optimizer=optz, loss='mse',metrics=['mse'])

nn.summary()
train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                    batch_size = batch_size ,epochs = epochs ,\
                    validation_data=(Xval, Yval), shuffle=True, callbacks=[early_stop])

# ================ Evaluate Model  ===========================

# learning curve
plt.figure(5)
plt.plot(train_hist.history['mse'])
plt.plot(train_hist.history['val_mse'])
plt.title('distance')
plt.ylabel('distance')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'validate'])
plt.savefig('./Results/hist_dist.png')


Ypredtrain = nn.predict(Xtrain)
Ypred = nn.predict(Xtest)
err = np.mean(abs(Ytest-Ypred),axis=0)**2
print( "--- MSE: --- %s" %(err))
print(np.corrcoef(Ytest[:,0], Ypred[:,0]))
print(np.corrcoef(Ytest[:,1], Ypred[:,1]))

# Histogramm of errors on test Data
plt.figure(4)
plt.hist(abs(Ytest[:, 0] - Ypred[:, 0]), bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('Estimate error')
plt.grid(True)
plt.title('histogram of estimation error mit.')
plt.savefig('./Results/hist_error_mit.png')

plt.figure(5)
plt.hist(abs(Ytest[:, 1] - Ypred[:, 1]), bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('Estimate error')
plt.grid(True)
plt.title('histogram of estimation error unmit.')
plt.savefig('./Results/hist_error_unmit.png')

# scatter plot of Y_true VS Y_pred
plt.figure(8)
plt.scatter(Ytest[:100, 0], Ypred[:100, 0], facecolors='none', edgecolors='b')
plt.scatter(Ytest[:100, 1], Ypred[:100, 1], facecolors='none', edgecolors='r')
plt.title(' est vs ground')
plt.ylabel('est')
plt.xlabel('ground')
plt.legend(['mitBER', 'umitBER'])
plt.grid(True)
plt.savefig('./Results/scatter_TF.png')


#============== save model to file =============
model_json = nn.to_json()
with open("./Results/model_TF.json", "w") as json_file:
    json_file.write(model_json)
    nn.save_weights("./Results/model_TF.h5")
    print("Saved model to disk")


# # save down pred data
# sio.savemat(dataID+'Ypred_TF.mat', {'Ypred':Ypred})
