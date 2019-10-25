#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Ber Predict with Linear Regression 
=========================================================


"""





import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import hdf5storage # load mat files
import scipy.io as sio

# Load the  dataset
#diabetes = datasets.load_diabetes()
dataID = './Data/'
t = hdf5storage.loadmat(dataID+'X16k.mat')  # XLong1    XScale1
X = t['X']
t = hdf5storage.loadmat(dataID+'Y16k.mat')  #  YLong1   YScale1
Y= t['Y']


Y = - 10*(np.log(Y/1000)/np.log(10));   # log value
for i in range(0,Y.shape[0]):
    for j in range(0,2):
        if np.isinf(Y[i,j]) or np.isnan(Y[i,j]):
            Y[i,j] = 40
            
ID = 1 ####################### 0 for unmit, 1 for mit (default)
            
Y = Y[:,ID]

# Split the data into training/testing sets
Xtrain = X[:-30000]
Xtest = X[-30000:]

# Split the targets into training/testing sets
Ytrain = Y[:-30000]
Ytest = Y[-30000:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(Xtrain, Ytrain)

# Make predictions using the testing set
Ypred = regr.predict(Xtest)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Ytest, Ypred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Ytest, Ypred))

print(np.corrcoef(Ytest, Ypred))

## Plot outputs
#plt.figure(1)
#plt.axis([0,30,0,30])
#plt.grid(True)
#plt.scatter( Ytest, Ypred, color='blue',facecolor='none')
#plt.title(' est vs ground')
#plt.ylabel('est')
#plt.xlabel('ground')
#plt.legend(['mitBER'])
#plt.savefig('./Results/reg_scatter_%d.png'%ID)



sio.savemat('./Results/Ytest-linReg-16k%d.mat'%ID, {'Ytest':Ytest}) 
sio.savemat('./Results/Ytest-linReg-16k%dn.mat'%ID, {'Ypred':Ypred}) 


#plt.plot(Ytest, Ypred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
