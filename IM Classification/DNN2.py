# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:28:13 2019

@author: Yue Xu
"""



from ToolBox import tools
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

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
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')


epochs = 300


t = hdf5storage.loadmat('XLong2.mat')
X = t['X']
add_type_len = len(X[:,0])
add_type = np.zeros((add_type_len,1))
for i in range(add_type_len):
    add_type[i,:] = 1

X = np.array(X)
X_full = np.column_stack((X,add_type))
X_full = np.reshape(X_full,(-1,10))

t = hdf5storage.loadmat('YLong2.mat')
Y = t['Y']

Y_unm = Y[:,0]/1000
Y_m = Y[:,1]/1000
Y_m = np.reshape(Y_m,(-1,1))
y_real = Y_m;
Y_m = tools.createClassfication(Y_m)
Y_m = np.reshape(Y_m,(-1,1))

Y_m = to_categorical(Y_m)

len1 = np.shape(X_full)
testLen = int( len1[0]*0.1)

batch_size = testLen  


x_test  = X_full[-2 * testLen:,:]
y_test  = Y_m[-2 * testLen:,:]
x_train = X_full[:-2 *testLen,:]
y_train = Y_m[:-2 *testLen,:]
y_qber_real = y_real[-2 * testLen:,:]


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(19, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, \
          nesterov=True)
# ‘categorical_crossentropy’
model.compile(loss='mse',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle = True,
          validation_split=0.1)

model.summary()

score = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
y_qber_est = tools.mapBERvalue(y_pred);
acc = np.sum(y_pred==y_test)*1.0/len(y_test)
avgqerr = tools.calculateQBER(y_qber_est,y_qber_real)
print( "--- acc %s ---" %(acc))
print( "--- avgqerr %s ---" %(avgqerr))


plt.figure(1)
plt.hist(y_qber_est,bins=200)
plt.figure(2)
plt.hist(y_qber_real,bins=200)
plt.figure(3)
plt.hist(y_test,bins=200)