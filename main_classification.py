"""

File description:

- This code is for training a NN as a CLASSIFIER for BER estimation. Right now the code runs
with 5 input features. The number of output neurons depends on the quantization levels (defined in the ToolBox
 file). So far, we are only considering unmitigated BER and FFT-based mitigated BER.

 To run the network on the unmitigated BER, set the flag_ber (line 43) to 0. To run it on the mitigated BER
 , set the flag_ber to 1.

- The code has no input. All it needs is the path to the data (line 51 and 60). It reads .mat files, as the
data was generated by Matlab. All the data needs to be kept in the folder named 'Data'.

- The code outputs:

1) It prints the averaged quantization error between the quantized true test BER and the quantized predicted test BER.
2) It prints the correlation coefficient (corr) between the quantized true test BER and the quantized predicted test BER.
3) It shows and saves the confusion matrix and the heat matrix
4) It also saves the model and its weights.

Note: All the saved files are in a folder named 'Results'.

"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from ToolBox import tools
import hdf5storage
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)
tf.random.set_seed(0)

# ========================= Parameter Settings ================
flag_ber = 1 #0 is unmit data; 1 is mit data
epochs = 50
batch_size = 512
dataID = './Data/'
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')  # Early stopping  for validation


# ========================= Loading and processing Data ================
t = hdf5storage.loadmat(dataID+'X16k1.mat')
X = t['X']
add_type_len = len(X[:,1])  # adding interference mitigation type to our features [0,1,2,3,4,5]
add_type = np.zeros((add_type_len,1))
for i in range(add_type_len):
    add_type[i,:] = 1
X = np.array(X)
X_full = np.column_stack((X,add_type))

t = hdf5storage.loadmat(dataID+'Y16k1.mat')
Y = t['Y']
Y_m = Y[:,flag_ber]/1000  # normalize mitigated BER (it was generated in matlab with a scale of 1000)
Y_m = np.reshape(Y_m,(-1,1))
y_real = Y_m # save the real BER value for later evaluation
Y_m = tools.createClassfication(Y_m) # quantize the BER
Y_m = np.reshape(Y_m,(-1,1))
Y_m = to_categorical(Y_m)  # convert the levels into class labels  [0-18]

# splitting into training and testing
len1 = np.shape(X_full)
testLen = int( len1[0]*0.2)
x_test  = X_full[-testLen:,:]
y_test  = Y_m[-testLen:,:]
x_train = X_full[:-testLen,:]
y_train = Y_m[:-testLen,:]
y_qber_real = y_real[-testLen:,:] # true BER to be used in testing (calculating quantization error)



#================ Model Building ===========================
model = Sequential()
model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(Y_m.shape[1], activation='softmax'))

#================ Model Compiling ===========================
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) # 'mse'
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle = True)
model.summary()


# ================ Evaluate Model  ===========================

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
y_qber_est = tools.mapBERvalue(y_pred)
# acc = np.sum(y_pred==y_test)*1.0/len(y_test)
avgqerr = tools.calculateQBER2(y_qber_est,y_qber_real)
# print( "--- acc %s ---" %(acc))
print( "--- avg_quantization_err %s ---" %(avgqerr))

corr = np.corrcoef(y_pred, y_test)[0][1]
print("--- corr(level) %s ---" %(corr))


# plt.figure(1)
# plt.hist(y_qber_est,bins=200)
# plt.xlabel('Estinated Ber Level (q)')
# plt.ylabel('Quantity')
# plt.savefig('./Results/histogram_estimated_BER.png')
#
#
# plt.figure(2)
# plt.hist(y_qber_real,bins=200)
# plt.xlabel('Real Ber Level (q)')
# plt.ylabel('Quantity')
# plt.savefig('./Results/histogram_real_BER.png')


cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(4)
plt.imshow(cm, cmap='binary')
plt.savefig('./Results/Confusion_matrix.png')

tools.cm_analysis(y_test, y_pred, ymap=None, figsize=(10,10))

#============== save model to file =============
model_json = model.to_json()
with open("./Results/model_classifer.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("./Results/model_classifer.h5")
    print("Saved model to disk")