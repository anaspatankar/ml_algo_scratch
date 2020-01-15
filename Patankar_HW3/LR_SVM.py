
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
import time


# In[12]:


trainingSet = pd.read_csv("trainingSet.csv")
testSet = pd.read_csv("testSet.csv")


# In[13]:


#Logistic Regression
def lr(trainingSet, testSet):
    
    
    #Clearing unnamed rows and adding intercept column in training set
    df_train = trainingSet
    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    intercept = np.ones(len(df_train))
    df_train['intercept'] = intercept

    dt = df_train['decision']
    df_train.drop(['decision'],axis = 1, inplace = True)
    df_train['decision'] = dt    
    
    #Clearing unnamed rows and adding intercept column in testing set
    df_test = testSet
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    intercept = np.ones(len(df_test))
    df_test['intercept'] = intercept

    dk = df_test['decision']
    df_test.drop(['decision'],axis = 1, inplace = True)
    df_test['decision'] = dk    
    
    X = df_train.iloc[:,:len(df_train.columns)-1]
    w = np.zeros(len(df_train.columns)-1)
    
    count = 1    

    while count < 501:   

        
        dot_product = np.dot(X, w)
        sigmoid = 1/(1+np.exp(-dot_product))
        addvalues = sigmoid - np.array(df_train['decision']) 
        gradient = np.dot(X.T,addvalues) + (0.01 * w)
        w_old = w
        
        learn_rate = 0.01 * gradient
        w = np.subtract(w, learn_rate)
    
        stop_factor = np.subtract(w,w_old)
        threshold = LA.norm(stop_factor)
        if threshold < 1e-6:
            break
        count = count + 1 
        

#Training Acciracy prediction

    
    prediction_train = []
    dot_product = np.dot(X, w)
    sigmoid = 1/(1+np.exp(-dot_product))
        
    for probability in sigmoid:
        if probability > 0.5:
            prediction_train.append(1)
        else:
            prediction_train.append(0)

    match = []
    for row in range(len(df_train)):
        if df_train['decision'].iloc[row] == prediction_train[row]:
            match.append(1)
        else:
            match.append(0)

    Accuracy_train = round(sum(match)/len(match), 2)
    
    print("Training Accuracy LR: " + str(round(Accuracy_train,2)))

# Testing Accuracy

    X_test = df_test.iloc[:,:len(df_train.columns)-1]
    dot_product = np.dot(X_test, w)

    prediction_prob = []
    prediction_test = []

    sigmoid = 1/(1+np.exp(-dot_product))
 
    for probability in sigmoid:
        if probability > 0.5:
            prediction_test.append(1)
        else:
            prediction_test.append(0)

    match = []
    for row in range(len(df_test)):
        if df_test['decision'].iloc[row] == prediction_test[row]:
            match.append(1)
        else:
            match.append(0)

    Accuracy_test = round(sum(match)/len(match), 2)

    print("Testing Accuracy LR: " + str(round(Accuracy_test,2)))
    
    #return Accuracy_test


# In[14]:


lr(trainingSet, testSet)


# In[17]:

#Support Vector Machines
def svm(trainingSet, testSet):
    
    df_train = trainingSet
    df_test = testSet

    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    df_train['decision'] = df_train['decision'].replace(0, -1)

    dt = df_train['decision']
    df_train.drop(['decision'],axis = 1, inplace = True)
    intercept = np.ones(len(df_train))
    df_train['intercept'] = intercept
    df_train['decision'] = dt

    df_test = testSet
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    df_test['decision'] = df_test['decision'].replace(0, -1)

    dk = df_test['decision']
    df_test.drop(['decision'],axis = 1, inplace = True)
    intercept = np.ones(len(df_test))
    df_test['intercept'] = intercept
    df_test['decision'] = dk

    X = df_train.iloc[:,:len(df_train.columns)-2]
    w_non_intercept = np.zeros(len(df_train.columns)-2)
    w_intercept = 0
    
    count = 1
    while count < 501:
 

        dot_product = np.dot(X, w_non_intercept)
        x = np.multiply(df_train.iloc[:,-1], dot_product)
    
        row_hingeloss = list((np.where(x<=1))[0])
        row_noloss = list((np.where(x>1))[0])
        
        #Evaluating the gradient for non_intercept columns
        non_intercept_value = 0.01 * np.tile(w_non_intercept, (len(df_train),1) )
        non_intercept_value[row_hingeloss] = non_intercept_value[row_hingeloss] - np.array(df_train.iloc[row_hingeloss,:(len(df_train.columns)-2)].multiply(df_train.iloc[row_hingeloss,-1], axis ="index"))
          
        #Evaluating the gradient for intercept column
        intercept_value = np.repeat(w_intercept, len(df_train))
        intercept_value[row_hingeloss] = -np.multiply(df_train.iloc[row_hingeloss,-2],(df_train.iloc[row_hingeloss,-1]))
        intercept_value_sum = intercept_value.sum(axis = 0)
        
        #Cumulating intercept and non intercept gradients
        sum_total = non_intercept_value.sum(axis = 0)
        sum_total = np.append(sum_total, intercept_value_sum)
        W_grad = sum_total/len(df_train)
        w_non_intercept = np.append(w_non_intercept, w_intercept)
        w = np.array(w_non_intercept)
        
        w_old = w 
        
        #Learning for w
        learn_rate = 0.5 * W_grad
        w = np.subtract(w, learn_rate)
        w_non_intercept = w[:-1]
        w_intercept = w[-1]

        #print(count)
    
        stop_factor = np.subtract(w,w_old)
        l_2 = LA.norm(stop_factor)
        if l_2 < 1e-6:
            break
        count = count + 1                     
    
    #Training Accuracy

    X = df_train.iloc[:,:len(df_train.columns)-1]
    training = list(np.dot(X, w))
    prediction_train = []
    for i in training:
        if i > 0:
            prediction_train.append(1)
        else:
            prediction_train.append(-1)
    
    match = []
    for row in range(len(df_train)):
        if df_train['decision'].iloc[row] == prediction_train[row]:
            match.append(1)
        else:
            match.append(0)

    Accuracy_train = round(sum(match)/len(match),2)

    print("Training Accuracy SVM: " + str(round(Accuracy_train,2)))
    
    #Testing Accuracy

    X_test = df_test.iloc[:,:len(df_train.columns)-1]
    testing = np.dot(X_test, w)
    prediction_test = []
    
    for i in testing:
        if i > 0:
            prediction_test.append(1)
        else:
            prediction_test.append(-1)
    
    match = []
    for row in range(len(df_test)):
        if df_test['decision'].iloc[row] == prediction_test[row]:
            match.append(1)
        else:
            match.append(0)

    Accuracy_test = round(sum(match)/len(match),2)

    print("Testing Accuracy SVM: " + str(round(Accuracy_test,2)))
    
    return Accuracy_test
    


# In[22]:


svm(trainingSet, testSet)


# In[19]:


def lr_svm(trainingSet, testSet, modelIdx):
    if modelIdx == 1:
        print(lr(trainingSet, testSet))
    elif modelIdx == 2:
        print(svm(trainingSet, testSet))


# In[23]:


lr_svm(trainingSet, testSet, 1)

