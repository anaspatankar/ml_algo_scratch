#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
np.random.seed(0)
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


# In[6]:


dx = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
df = pd.DataFrame()
for i in range(10):
    df = df.append(dx[dx['label'] == i][:10], ignore_index = True)
X = df[['dim1', 'dim2']]

Z_single = linkage(X, method='single', metric='euclidean', optimal_ordering=False)
Z_complete = linkage(X, method='complete', metric='euclidean', optimal_ordering=False)
Z_average = linkage(X, method='average', metric='euclidean', optimal_ordering=False)


# In[7]:


def hierarchial(type, K):
    
    dx = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
    
    df = pd.DataFrame()
    
    for i in range(10):
        df = df.append(dx[dx['label'] == i][:10], ignore_index = True)
    X = df[['dim1', 'dim2']]
    
    cluster =  fcluster(type, K, criterion='maxclust')
    df['cluster']= cluster
    
    
    
    class_count = df.groupby('label').count()['cluster']
    cluster_count = df.groupby('cluster').count()['label']
    
    class_entropy = sum(-np.multiply(np.array(class_count/len(df)), np.log(np.array(class_count/len(df)))))
    cluster_entropy = sum(-np.multiply(np.array(cluster_count/len(df)), np.log(np.array(cluster_count/len(df)))))
    
    
    N = len(df)
    class_num = sorted(list(df['label'].unique()))
    cluster_num = sorted(list(df['cluster'].unique()))
    prob_class = {i:(len(df[df['label']==i])/N) for i in class_num}
    prob_clusters = {i:(len(df[df['cluster']==i])/N) for i in cluster_num}
    prob_cg = {i: {j:(len(df[(df['label']==i) & (df['cluster']==j)])/N) for j in cluster_num} for i in class_num}
    
    list1 = []
    
    for i in range(10):
        X = np.array([prob_cg[i][cnum]*(np.log(prob_cg[i][cnum]/(prob_class[i] * prob_clusters[cnum]))) for cnum in np.linspace(1,K,K, dtype = int)])
        X = X[~np.isnan(X)]
        X[X<0] = 0
        Y = np.sum(X)
        list1.append(Y)
    
    NMI = np.sum(list1)/(class_entropy+cluster_entropy)
    
    return NMI


# In[8]:


NMI_single = hierarchial(Z_single, 8)
NMI_complete = hierarchial(Z_complete, 8)
NMI_average = hierarchial(Z_average, 8)


# In[9]:


print("NMI for single linkage: " + str(NMI_single))
print("NMI for complete linkage: " + str(NMI_complete))
print("NMI for average linkage: " + str(NMI_average))


# In[10]:


NMI = [0.3179, 0.356, 0.392, 0.379]
Type = ["kmeans", "z_single", "z_complete", "z_average"]
sns.barplot(Type, NMI)


# In[ ]:




