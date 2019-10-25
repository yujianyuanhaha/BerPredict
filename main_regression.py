import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import hdf5storage # load mat files
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Flatten, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import scipy.io as sio
import math

np.random.seed(0)
tf.random.set_seed(0)

# ======================== Parameters ========================

dataID = './Data/'
epochs = 100   # number of learning epochs
batch_size = 512 #512
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto') # Early stopping

#  =============== load data =================================

t = hdf5storage.loadmat(dataID+'X160k_shuffled.mat')  #   X_shuffled_1
X = t['X']
t = hdf5storage.loadmat(dataID+'Y160k_shuffled.mat')  #   Y_shuffled_1
Y= t['Y']
Y= Y[:,1]
# ================== Data processing ###################

# back to BER from 0 to 1
sc_factor = 1000
Y = Y/sc_factor

# ================  split data =================================

train_fraction = 0.7
train_size = int(train_fraction*Y.shape[0])
Xtrain = X[:train_size,:]
Ytrain = Y[:train_size]


val_fraction = 0.5
val_size = int((Y.shape[0] - train_size)*val_fraction)
Xval = X[train_size:train_size+val_size, :]
Yval = Y[train_size:train_size+val_size]

Xtest = X[train_size+val_size:,:]
Ytest = Y[train_size+val_size:]



# extend dim
Xtrain = Xtrain[:, :, np.newaxis]
Xtest = Xtest[:, :, np.newaxis]
Xval = Xval[:, :, np.newaxis]


Ytest = Ytest[:,np.newaxis]
Ytrain = Ytrain[:,np.newaxis]
Yval = Yval[:,np.newaxis]

# make sure data has same distribution

plt.figure(1)
plt.hist(Ytrain[:,0],bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('Y')
plt.grid(True)
plt.title('histogram of BER_train mit.')
plt.savefig('./Results/hist_BER_test_mit.png')
#
# plt.figure(2)
# plt.hist(Ytrain[:,1],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y')
# plt.grid(True)
# # plt.title('histogram of BER_train mit.')
# # plt.savefig('./Results/hist_BER_test_mit.png')
#
#
#
#
# plt.figure(1)
# plt.hist(Ytest[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y')
# plt.grid(True)
# # plt.title('histogram of BER_test umit.')
# plt.savefig('./Results/hist_BER_test_umit.png')
#
# plt.figure(2)
# plt.hist(Ytest[:,1],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y')
# plt.grid(True)
# # plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_BER_test_mit.png')
#
#
#
#
# plt.figure(1)
# plt.hist(Yval[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y')
# plt.grid(True)
# # plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_BER_Y_unmit.png')
#
# plt.figure(2)
# plt.hist(Yval[:,1],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('Y')
# plt.grid(True)
# # plt.title('histogram of BER_test umit.')
# plt.savefig('./Results/hist_BER_Y_mit.png')
#
#
#
# plt.figure(3)
# plt.hist(Xtrain[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('X_feature_1')
# plt.grid(True)
# # plt.title('histogram of BER_test mit.')
# # plt.savefig('./Results/hist_BER_val_mit1.png')
#
# plt.figure(3)
# plt.hist(Xtest[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('X_feature_1')
# plt.grid(True)
# # plt.title('histogram of BER_test mit.')
# # plt.savefig('./Results/hist_BER_val_mit1.png')
#
#
# plt.figure(3)
# plt.hist(Xval[:,0],bins=64)
# plt.ylabel('Number of occurence')
# plt.xlabel('X_feature_1')
# plt.grid(True)
# # plt.title('histogram of BER_test mit.')
# plt.savefig('./Results/hist_X.png')




#================ Extra Functions ===========================


# def dist(y_true, y_pred):
#      return tf.reduce_mean(tf.sqrt(tf.square(tf.abs(y_pred-y_true)) ))

# def clip(x,fact=sc_factor):
#     return tf.keras.backend.clip(x,0.0,1000.0/fact)
#
# def scale(x,fact=1000):
#     return x*fact

#================ Model Building ===========================

nn_input  = Input((X.shape[1],1))
nn_output = Flatten()(nn_input)
nn_output = Dense(512,activation='tanh')(nn_output)
nn_output = Dense(512,activation='tanh')(nn_output)
nn_output = Dense(512,activation='tanh')(nn_output)
nn_output = Dense(1,activation='sigmoid')(nn_output)
# nn_output = Lambda(scale)(nn_output)
nn = Model(inputs=nn_input,outputs=nn_output)

#================ Model Compiling ===========================


optz = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, amsgrad=False)
nn.compile(optimizer=optz, loss='mse',metrics=['mse'])
nn.summary()
train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                    batch_size = batch_size ,epochs = epochs ,\
                    validation_data=(Xval, Yval), shuffle=True, callbacks=[early_stop])

# ================ Evaluate Model  ===========================

# printing the test MSE and correlation coefficient
Ypred = nn.predict(Xtest)
err = np.mean(abs(Ytest-Ypred),axis=0)
err_db = -10*math.log(err,10)
print(err_db)
print( "--- MSE: --- %s" %(err))
print(np.corrcoef(Ytest[:,0], Ypred[:,0]))


# learning curve plot
plt.figure(4)
plt.plot(train_hist.history['mse'])
plt.plot(train_hist.history['val_mse'])
plt.title('distance')
plt.ylabel('distance')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'validate'])
plt.savefig('./Results/learning_curve.png')




# Histogram of errors on test Data

plt.figure(5)
plt.hist(abs(Ytest[:, 0] - Ypred[:, 0]), bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('Estimate error')
plt.grid(True)
plt.title('histogram of estimation error mit.')
plt.savefig('./Results/hist_error_mit.png')


# scatter plot of Y_true VS Y_pred
plt.figure(6)
plt.scatter(Ytest[:, 0], Ypred[:, 0], facecolors='none', edgecolors='b')
plt.title(' est vs ground')
plt.ylabel('est')
plt.xlabel('ground')
plt.legend([ 'mitBER'])
plt.grid(True)
plt.savefig('./Results/scatter_true_VS_predict.png')


# scatter plot of Y_true VS Y_pred in dB
Y_test_dB = -10*(np.log(Ytest)/np.log(10))
Y_pred_dB = -10*(np.log(Ypred)/np.log(10))
plt.figure(7)
plt.scatter(Y_test_dB[:, 0], Y_pred_dB[:, 0], facecolors='none', edgecolors='b')
plt.title(' est vs ground')
plt.ylabel('est_dB')
plt.xlabel('ground_dB')
plt.legend([ 'mitBER'])
plt.grid(True)
plt.savefig('./Results/scatter_true_VS_predict_dB.png')


# scatter plot of Y_true VS estimation error
plt.figure(8)
plt.scatter(Ytest[:, 0], abs(Ytest[:, 0] - Ypred[:, 0]), facecolors='none', edgecolors='b')
plt.title(' ground vs error')
plt.ylabel('error')
plt.xlabel('ground')
plt.legend(['mitBER'])
plt.grid(True)
plt.savefig('./Results/scatter_true_VS_error_all.png')

#============== save model to file =============
model_json = nn.to_json()
with open("./Results/model_TF.json", "w") as json_file:
    json_file.write(model_json)
    nn.save_weights("./Results/model_TF.h5")
    print("Saved model to disk")
