"""

File description:

- This code defines helper function for the NN classifier

- The main functions here are:

1) createClassfication

Has an input the BER vector (in [0,1] scale) and converts it to a class label based on a given quantization map

2) Quantization_map

Defines the quantization levels (range)

3) mapBERvalue

Takes vector of class labels and maps them to the corresponding quantized level

4) calculateQBER2

Calculates Quantization error between the true and predicted quantized levels

5) cm_analysis

Confusion matrix calculation and saving

"""

import numpy as np
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class tools: 
    def createClassfication(Y):  # takes real value BER vector and converts it to a class label
        
        y_classified = []
        lengthY = len(Y)
        
        for i in range(lengthY):
            temp_value = Y[i,0]
            temp_y_type = tools.classficationType(temp_value)
            y_classified.append(temp_y_type)

        return y_classified

    def classficationType(y):  # takes one real value BER to the class label

        ber_quantization_map = tools.Quantization_map(5) # defines the range of possible quantized BER levels, e.g., 10^-4 till 1

        # from quantization levels to classes
        if y >= 0.9:
            y_class = 0
        else:
            y_level = ber_quantization_map < y
            idx = np.where(y_level == False)
            y_class = int(idx[0][0]) + 1

        return y_class

    def Quantization_map(minBer): # defines the quantization levels (range)

        ber_map = []
        for i in range(minBer):
            temp_ber_level = 1 / np.power(10, minBer - i)
            for j in range(9):
                temp_ber_start = (j + 1) * temp_ber_level
                ber_map.append(temp_ber_start)

        return ber_map

    def mapBERvalue(Y):  # takes vector of class labels and maps them to the corresponding quantized level
        
        y_mv = []
        lengthY = len(Y)
        
        for i in range(lengthY):
            temp_value = Y[i]
            temp_y_type = tools.creatBERvalue(temp_value)
            y_mv.append(temp_y_type)

        return y_mv            

    def creatBERvalue(Y): # takes a class label and converts it to quantized BER value

        ber_map = tools.Quantization_map(5)
        
        if Y==0:
            y = 0.95
        else:
            tempIdx = Y-1
            y = (ber_map[tempIdx-1] + ber_map[tempIdx])/2
            
        return y


    # def calculateQBER(est,real):  ### Calculates Quntization error
    #
    #     sumBer = 0
    #     lengthY = len(est)
    #     for i in range(lengthY):
    #         err = abs(est[i] - real[i])
    #         sumBer = sumBer + err
    #
    #     ber = -10*math.log(sumBer / lengthY,10)
    #     return ber

    def calculateQBER2(est,real): ### Calculates Quntization error
        sumBer = 0
        lengthY = len(est)
        
        for i in range(lengthY):
            temp_est = est[i]
            temp_real = real[i,0]
            if temp_est <= 0:
                temp_est = 1/np.power(10,9)
            if temp_real <=0:
                temp_real = 1/np.power(10,9)
            temp = abs(-10*math.log(temp_est,10) - -10*math.log(temp_real,10))
            sumBer = sumBer + temp
        ber = sumBer / lengthY
        return ber

    def cm_analysis(y_true, y_pred, ymap=None, figsize=(10,10)):  # confusion matrix calculation and saving
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
        cm = confusion_matrix(y_true, y_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        np.save("./Results/cm_perc_m.npy",cm_perc)
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='coolwarm')
        plt.savefig('./Results/confusion_heat_matrix.png')
        plt.show()