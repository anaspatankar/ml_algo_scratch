
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("trainingSet.csv")

import pandas as pd

df = pd.read_csv("trainingSet.csv")


def nbc(t_frac=1):

    #loading train data
    df = pd.read_csv("trainingSet.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df = df.sample(frac=t_frac, random_state=47)

    #Prior probabilities
    Yes= df[(df.iloc[:,52] == 1)]['decision'].count()
    No = df[(df.iloc[:,52] == 0)]['decision'].count()
    Total= df['decision'].count()
    Prior_1 = Yes/Total
    Prior_0 = No/Total
    

    #loading probabilities of different cases as a dictionary for 0 and 1

    dict_1 = {}  
    for i in range(0,52):
        bins = list(df.iloc[:,i].unique())
        p = {}
        for k in bins:
            Yes_i_variable= df[(df.iloc[:,52] == 1) & (df.iloc[:, i] == k)].iloc[:,i].count()
            p[k] = Yes_i_variable/Yes
        dict_1[i] = p

    
    dict_0 = {}  
    for i in range(0,52):
        bins = list(df.iloc[:,i].unique())
        p = {}
        for k in bins:
            No_i_variable= df[(df.iloc[:,52] == 0) & (df.iloc[:, i] == k)].iloc[:,i].count()
            p[k] = No_i_variable/No
        dict_0[i] = p



        #conditional probability
        def con_prob(row_num, decision):
            position = df.iloc[row_num]
            prob = 1
            if (decision == 1):
                for i in range(0, 52):
                    bins = position[i]
                    try:
                        prob = prob * float(dict_1.get(i).get(bins))
                    except:
                        prob = prob
            elif (decision == 0):
                for i in range(0, 52):
                    bins = position[i]
                    try:
                        prob = prob * float(dict_0.get(i).get(bins))
                    except:
                        prob = prob

            return (prob)

    #TRAIN DATA##
    pred = []
    for n in range(0,len(df)):
        numS = con_prob(row_num=n, decision = 1)*Prior_1
        den = numS+(con_prob(row_num=n, decision = 0)*Prior_0)
        Success = numS/den
        numF= con_prob(row_num=n, decision = 0)*Prior_0
        Failure = numF/den
        if Success > Failure:
            pred.append(1)
        else:
            pred.append(0)
    df['MyPred'] = pred
    matches1 = df[(df['decision'] == df['MyPred'])]['MyPred'].count()
    
    Training_Accuracy = round(matches1/len(df),2)
    


    
    df= pd.read_csv("testSet.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    #TESTING
    pred = []
    for n in range(0,len(df)):
        numS = con_prob(row_num=n, decision = 1)*Prior_1
        den = numS+(con_prob(row_num=n, decision = 0)*Prior_0)
        Success = numS/den
        numF= con_prob(row_num=n, decision = 0)*Prior_0
        Failure = numF/den
        if Success > Failure:
            pred.append(1)
        else:
            pred.append(0)
    df['MyPred'] = pred
    matches = df[(df['decision'] == pred)]['MyPred'].count()
    
    Test_Accuracy = round(matches/len(df),2)
    return (Training_Accuracy, Test_Accuracy)


nbc()

#binning continous variables and then checking accuracy
def split(bins):
    df = pd.read_csv("dating.csv")
    processed_columns = ['gender', 'race_o', 'samerace', 'race','field', 'decision']
    processed_id = []
    for i in processed_columns:
        processed_id.append(df.columns.get_loc(i))
    
    col_id = list(np.arange(0,52))
    continuous_id = [x for x in col_id if x not in processed_id]
    
    for i in continuous_id:
        processed = pd.cut(df.iloc[:, i],bins=bins,labels=list(range(0,bins)))
        df.iloc[:, i] = processed
        num_bins = []
        for j in range(0,bins):
            num_bins.append(len(df[df.iloc[:, i] == j]))
    testSet = df.sample(frac=0.2, random_state=47)
    trainingSet = df.drop(testSet.index)
    testSet.to_csv('testSet.csv',index=False)
    trainingSet.to_csv('trainingSet.csv',index=False )

bins=[2,5,10,50,100,200]
train_a=[]
test_a=[]
for i in bins:
    split(i)
    Training_Accuracy,Testing_Accuracy=nbc()
    train_a.append(Training_Accuracy)
    test_a.append(Testing_Accuracy)


x = bins
y1 = train_a
y2 = test_a
plt.plot(x, y1,color='b',label="Train")
plt.plot(x, y2,color='g',label="Test")
plt.legend()
plt.xlabel('Bins')
plt.ylabel('Accuracy')
plt.title('Acc vs Bins')

