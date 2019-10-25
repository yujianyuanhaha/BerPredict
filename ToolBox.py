
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class tools: 
    def createClassfication(Y):  # takes real value BER and quantize it
        
        y_classified = []
        
        lengthY = len(Y)
        
        for i in range(lengthY):
            temp_value = Y[i,0]
            temp_y_type = tools.classficationType3(temp_value)
            y_classified.append(temp_y_type)
            
            
        return y_classified
            

    def classficationType3(y):  # Quantization levels and map
        
        y_type = 0

        min_ber = 1/np.power(10,4)
        middle_ber21 = 3 * 1/np.power(10,4)
        middle_ber22 = 5 * 1/np.power(10,4)
        middle_ber23 = 7 * 1/np.power(10,4)
        middle_ber24 = 9 * 1/np.power(10,4)
        middle_ber2 = 1/np.power(10,3)
        middle_ber31 = 3 * 1/np.power(10,3)
        middle_ber32 = 5 * 1/np.power(10,3)
        middle_ber33 = 7 * 1/np.power(10,3)
        middle_ber34 = 9 * 1/np.power(10,3)
        middle_ber3 = 1/np.power(10,2)
        middle_ber41 = 3 * 1/np.power(10,2)
        middle_ber42 = 5 * 1/np.power(10,2)
        middle_ber43 = 7 * 1/np.power(10,2)
        middle_ber44 = 9 * 1/np.power(10,2)
        max_ber = 1/np.power(10,1)
        middle_ber51 = 3.5 * 1/np.power(10,1)
        
        
        if y <= min_ber:
            y_type = 1
        elif y> min_ber and y <= middle_ber2:
            if y<=middle_ber21:
                y_type = 2
            elif y<=middle_ber22:
                y_type = 3
            elif y<=middle_ber23:
                y_type = 4
            elif y<=middle_ber24:
                y_type = 5
            else:
                y_type = 6
        elif y> middle_ber2 and y <= middle_ber3:
            if y<=middle_ber31:
                y_type = 7
            elif y<=middle_ber32:
                y_type = 8
            elif y<=middle_ber33:
                y_type = 9
            elif y<=middle_ber34:
                y_type = 10
            else:
                y_type = 11

        elif y> middle_ber3 and y <= max_ber:
            if y<=middle_ber41:
                y_type = 12
            elif y<=middle_ber42:
                y_type = 13
            elif y<=middle_ber43:
                y_type = 14
            elif y<=middle_ber44:
                y_type = 15
            else:
                y_type = 16

        elif y> max_ber and y <= middle_ber51:
            y_type = 17
            
        elif y> middle_ber51 and y<=1:
            y_type = 18   
            
        elif y>1:
            y_type = 0
                            
        return y_type


    def mapBERvalue(Y):  # takes a class label and maps to the corresponding quantized level
        
        y_mv = []
        
        lengthY = len(Y)
        
        for i in range(lengthY):
            temp_value = Y[i]
            temp_y_type = tools.creatBERvalue3(temp_value)
            y_mv.append(temp_y_type)
            
            
        return y_mv



    def creatBERvalue3(Y):
        
        y = 0
        
        if Y == 1:
            y = 1/np.power(10,4)
        elif Y == 2:
            y = 2/np.power(10,4)
        elif Y == 3:
            y = 4/np.power(10,4)        
        elif Y == 4:
            y = 6/np.power(10,4)
        elif Y == 5:
            y = 8/np.power(10,4)            
        elif Y == 6:
            y = 9.5/np.power(10,4)
        elif Y == 7:
            y = 2/np.power(10,3)
        elif Y == 8:
            y = 4/np.power(10,3)
        elif Y == 9:
            y = 6/np.power(10,3)
        elif Y == 10:
            y = 8/np.power(10,3)
        elif Y == 11:
            y = 9.5/np.power(10,3)
        elif Y == 12:
            y = 2/np.power(10,2)
        elif Y == 13:
            y = 4/np.power(10,2)
        elif Y == 14:
            y = 6/np.power(10,2)
        elif Y == 15:
            y = 8/np.power(10,2)
        elif Y == 16:
            y = 9.5/np.power(10,2)
        elif Y == 17:
            y = 2/np.power(10,1)
        elif Y == 18:
            y = 5/np.power(10,1)
        elif Y == 0:
            y = 1
            
        return y


    def calculateQBER(est,real):  ### Calculates Quntization error

        sumBer = 0
        lengthY = len(est)
        
        for i in range(lengthY):
            err = abs(est[i] - real[i])
            sumBer = sumBer + err

            # temp_est = est[i]
            # temp_real = real[i]
            # temp = abs(-10*math.log(temp_est,10) - -10*math.log(temp_real,10))
            # sumBer = sumBer + temp

        ber = -10*math.log(sumBer / lengthY,10)
            
        return ber


    def cm_analysis(y_true, y_pred, ymap=None, figsize=(10,10)):
        """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args:
          y_true:    true label of the data, with shape (nsamples,)
          y_pred:    prediction of the data, with shape (nsamples,)
          filename:  filename of figure file to save
          labels:    string array, name the order of class labels in the confusion matrix.
                     use `clf.classes_` if using scikit-learn models.
                     with shape (nclass,).
          ymap:      dict: any -> string, length == nclass.
                     if not None, map the labels & ys to more understandable strings.
                     Caution: original y_true, y_pred and labels must align.
          figsize:   the size of the figure plotted.
        """
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
        cm = confusion_matrix(y_true, y_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
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
        #plt.savefig(filename)
        plt.show()