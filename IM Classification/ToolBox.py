

import numpy as np
import math

class tools: 
    def createClassfication(Y):
        
        y_classified = [];
        
        lengthY = len(Y);
        
        for i in range(lengthY):
            temp_value = Y[i,0]
            temp_y_type = tools.classficationType3(temp_value)
            y_classified.append(temp_y_type)
            
            
        return y_classified
            
    
            
        

    def classficationType(y):
        
        y_type = 0;
        
        min_ber = 1/np.power(10,5); #1
        middle_ber11 = 2 * 1/np.power(10,5);
        middle_ber12 = 4 * 1/np.power(10,5);
        middle_ber13 = 6 * 1/np.power(10,5);
        middle_ber14 = 8 * 1/np.power(10,5);
        middle_ber1 = 1/np.power(10,4);#6
        middle_ber21 = 2 * 1/np.power(10,4);
        middle_ber22 = 4 * 1/np.power(10,4);
        middle_ber23 = 6 * 1/np.power(10,4);
        middle_ber24 = 8 * 1/np.power(10,4);
        middle_ber2 = 1/np.power(10,3);
        middle_ber31 = 2 * 1/np.power(10,3);
        middle_ber32 = 4 * 1/np.power(10,3);
        middle_ber33 = 6 * 1/np.power(10,3);
        middle_ber34 = 8 * 1/np.power(10,3);
        middle_ber3 = 1/np.power(10,2);
        middle_ber41 = 2 * 1/np.power(10,2);
        middle_ber42 = 4 * 1/np.power(10,2);
        middle_ber43 = 6 * 1/np.power(10,2);
        middle_ber44 = 8 * 1/np.power(10,2);
        max_ber = 1/np.power(10,1);
        middle_ber51 = 2 * 1/np.power(10,1);
        middle_ber52 = 4 * 1/np.power(10,1);        
        
        
        if y <= min_ber:
            y_type = 1
            
        if y> min_ber and y <= middle_ber1:
            if y<=middle_ber11:
                y_type = 2
            else:
                if y<=middle_ber12:
                    y_type = 3
                else:
                    if y<-middle_ber13:
                        y_type = 4
                    else:
                        if y<middle_ber14:
                            y_type = 5
                        else:
                            y_type = 6
                            
        if y> middle_ber1 and y <= middle_ber2:
            if y<=middle_ber21:
                y_type = 7
            else:
                if y<=middle_ber22:
                    y_type = 8
                else:
                    if y<-middle_ber23:
                        y_type = 9
                    else:
                        if y<middle_ber24:
                            y_type = 10
                        else:
                            y_type = 11                    
                    

        if y> middle_ber2 and y <= middle_ber3:
            if y<=middle_ber31:
                y_type = 12
            else:
                if y<=middle_ber32:
                    y_type = 13
                else:
                    if y<-middle_ber33:
                        y_type = 14
                    else:
                        if y<middle_ber34:
                            y_type = 15
                        else:
                            y_type = 16

        if y> middle_ber3 and y <= max_ber:
            if y<=middle_ber41:
                y_type = 17
            else:
                if y<=middle_ber42:
                    y_type = 18
                else:
                    if y<-middle_ber43:
                        y_type = 19
                    else:
                        if y<middle_ber44:
                            y_type = 20
                        else:
                            y_type = 21

        if y> max_ber and y <= middle_ber51:
            y_type = 22
            
        if y> middle_ber51 and y <= middle_ber52:
            y_type = 23        

        if y> middle_ber52 and y<=1:
            y_type = 24   
            
        if y>1:
            y_type = 0
                            
        return y_type

    def classficationType2(y):
        
        y_type = 0;
        
        min_ber = 1/np.power(10,5); #1
        middle_ber11 = 3.5 * 1/np.power(10,5);
        middle_ber12 = 7 * 1/np.power(10,5);
        middle_ber1 = 1/np.power(10,4);#6
        middle_ber21 = 3.5 * 1/np.power(10,4);
        middle_ber22 = 7 * 1/np.power(10,4);
        middle_ber2 = 1/np.power(10,3);
        middle_ber31 = 3.5 * 1/np.power(10,3);
        middle_ber32 = 7 * 1/np.power(10,3);
        middle_ber3 = 1/np.power(10,2);
        middle_ber41 = 3.5 * 1/np.power(10,2);
        middle_ber42 = 7 * 1/np.power(10,2);
        max_ber = 1/np.power(10,1);
        middle_ber51 = 3.5 * 1/np.power(10,1);        
        
        
        if y <= min_ber:
            y_type = 1
            
        if y> min_ber and y <= middle_ber1:
            if y<=middle_ber11:
                y_type = 2
            else:
                if y<=middle_ber12:
                    y_type = 3
                else:
                    y_type = 4

                            
        if y> middle_ber1 and y <= middle_ber2:
            if y<=middle_ber21:
                y_type = 5
            else:
                if y<=middle_ber22:
                    y_type = 6
                else:
                    y_type = 7                   
                    

        if y> middle_ber2 and y <= middle_ber3:
            if y<=middle_ber31:
                y_type = 8
            else:
                if y<=middle_ber32:
                    y_type = 9
                else:
                    y_type = 10

        if y> middle_ber3 and y <= max_ber:
            if y<=middle_ber41:
                y_type = 11
            else:
                if y<=middle_ber42:
                    y_type = 12
                else:
                    y_type = 13

        if y> max_ber and y <= middle_ber51:
            y_type = 14
            
        if y> middle_ber51 and y<=1:
            y_type = 15   
            
        if y>1:
            y_type = 0
                            
        return y_type

    def classficationType3(y):
        
        y_type = 0;

        min_ber = 1/np.power(10,4);#6
        middle_ber21 = 3 * 1/np.power(10,4);
        middle_ber22 = 5 * 1/np.power(10,4);
        middle_ber23 = 7 * 1/np.power(10,4);
        middle_ber24 = 9 * 1/np.power(10,4);
        middle_ber2 = 1/np.power(10,3);
        middle_ber31 = 3 * 1/np.power(10,3);
        middle_ber32 = 5 * 1/np.power(10,3);
        middle_ber33 = 7 * 1/np.power(10,3);
        middle_ber34 = 9 * 1/np.power(10,3);
        middle_ber3 = 1/np.power(10,2);
        middle_ber41 = 3 * 1/np.power(10,2);
        middle_ber42 = 5 * 1/np.power(10,2);
        middle_ber43 = 7 * 1/np.power(10,2);
        middle_ber44 = 9 * 1/np.power(10,2);
        max_ber = 1/np.power(10,1);
        middle_ber51 = 3.5 * 1/np.power(10,1);        
        
        
        if y <= min_ber:
            y_type = 1
            
        if y> min_ber and y <= middle_ber2:
            if y<=middle_ber21:
                y_type = 2
            else:
                if y<=middle_ber22:
                    y_type = 3
                else:
                    if y<=middle_ber23:
                        y_type = 4
                    else:
                        if y<=middle_ber24:
                            y_type = 5
                        else:
                            y_type = 6

                            
        if y> middle_ber2 and y <= middle_ber3:
            if y<=middle_ber31:
                y_type = 7
            else:
                if y<=middle_ber32:
                    y_type = 8
                else:
                    if y<=middle_ber33:
                        y_type = 9
                    else:
                        if y<=middle_ber34:
                            y_type = 10
                        else:
                            y_type = 11                 
                    

        if y> middle_ber3 and y <= max_ber:
            if y<=middle_ber41:
                y_type = 12
            else:
                if y<=middle_ber42:
                    y_type = 13
                else:
                    if y<=middle_ber43:
                        y_type = 14
                    else:
                        if y<=middle_ber44:
                            y_type = 15
                        else:
                            y_type = 16  



        if y> max_ber and y <= middle_ber51:
            y_type = 17
            
        if y> middle_ber51 and y<=1:
            y_type = 18   
            
        if y>1:
            y_type = 0
                            
        return y_type




# In[2]:

    def mapBERvalue(Y):
        
        y_mv = [];
        
        lengthY = len(Y);
        
        for i in range(lengthY):
            temp_value = Y[i]
            temp_y_type = tools.creatBERvalue3(temp_value)
            y_mv.append(temp_y_type)
            
            
        return y_mv

    def creatBERvalue2(Y):
        
        y = 0;
        
        if Y == 1:
            y = 1/np.power(10,5)
        if Y == 2:
            y = 2/np.power(10,5)        
        if Y == 3:
            y = 5/np.power(10,5)        
        if Y == 4:
            y = 9/np.power(10,5)
            
            
        if Y == 5:
            y = 2/np.power(10,4)            
        if Y == 6:
            y = 5/np.power(10,4)
        if Y == 7:
            y = 9/np.power(10,4)
            
            
        if Y == 8:
            y = 2/np.power(10,3)            
        if Y == 9:
            y = 5/np.power(10,3)
        if Y == 10:
            y = 9/np.power(10,3)
            
            
        if Y == 11:
            y = 2/np.power(10,2)
        if Y == 12:
            y = 5/np.power(10,2)
        if Y == 13:
            y = 9/np.power(10,2)
            
        if Y == 14:
            y = 2/np.power(10,1)
        if Y == 15:
            y = 5/np.power(10,1)
            
        if Y == 0:
            y = 1
            
        return y


    def creatBERvalue3(Y):
        
        y = 0;
        
        if Y == 1:
            y = 1/np.power(10,4)
        if Y == 2:
            y = 2/np.power(10,4)        
        if Y == 3:
            y = 4/np.power(10,4)        
        if Y == 4:
            y = 6/np.power(10,4)
        if Y == 5:
            y = 8/np.power(10,4)            
        if Y == 6:
            y = 9.5/np.power(10,4)
            
            
            
        if Y == 7:
            y = 2/np.power(10,3)
        if Y == 8:
            y = 4/np.power(10,3)            
        if Y == 9:
            y = 6/np.power(10,3)
        if Y == 10:
            y = 8/np.power(10,3)
        if Y == 11:
            y = 9.5/np.power(10,3)
            
            
        if Y == 12:
            y = 2/np.power(10,2)
        if Y == 13:
            y = 4/np.power(10,2)           
        if Y == 14:
            y = 6/np.power(10,2)
        if Y == 15:
            y = 8/np.power(10,2)
        if Y == 16:
            y = 9.5/np.power(10,2)
            
        if Y == 17:
            y = 2/np.power(10,1)
        if Y == 18:
            y = 5/np.power(10,1)        
            
        if Y == 0:
            y = 1
            
        return y




    def calculateQBER(est,real):
        
        ber = 0;
        sumBer = 0
        
        lengthY = len(est);
        
        for i in range(lengthY):
            temp_est = est[i]
            temp_real = real[i]
            temp = abs(-10*math.log(temp_est,10) - -10*math.log(temp_real,10))
            sumBer = sumBer + temp

        ber = sumBer / lengthY   
            
        return ber
