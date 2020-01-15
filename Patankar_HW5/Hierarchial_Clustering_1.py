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


# In[2]:


dx = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)


# In[3]:


df = pd.DataFrame()
for i in range(10):
    df = df.append(dx[dx['label'] == i][:10], ignore_index = True)
X = df[['dim1', 'dim2']]


# In[4]:


Z_single = linkage(X, method='single', metric='euclidean', optimal_ordering=False)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(
    Z_single,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[5]:


Z_complete = linkage(X, method='complete', metric='euclidean', optimal_ordering=False)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(
    Z_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[17]:


Z_average = linkage(X, method='average', metric='euclidean', optimal_ordering=False)
plt.figure(figsize=(21, 7))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(
    Z_average,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# Single linkage

# In[7]:


def hierarchial(type, K):
    
    dx = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
    
    df = pd.DataFrame()
    
    for i in range(10):
        df = df.append(dx[dx['label'] == i][:10], ignore_index = True)
    X = df[['dim1', 'dim2']]
    
    cluster =  fcluster(type, K, criterion='maxclust')
    df['cluster']= cluster
    
    vertex = list(zip(df['dim1'], df['dim2']))
    s = df.groupby('cluster').mean()[['dim1', 'dim2']]
    centroid = [(s.loc[i]['dim1'], s.loc[i]['dim2']) for i in np.linspace(1,K,K, dtype = int)]
    
    def index_cluster(cluster_number):
        return (np.where(cluster == cluster_number)[0])
    clust_points = {}

    for num in np.linspace(1,K,K, dtype = int):
        clust_points[num] = [vertex[i] for i in index_cluster(num)]
        
    list_ss = []
    for clust_index in np.linspace(1,K,K, dtype = int):
        for i in range(len(clust_points[clust_index])):
            list_ss.append(np.square(distance.euclidean(centroid[clust_index-1],clust_points[clust_index][i])))
    WC_SSD = sum(list_ss)
    #print(WC_SSD)
    
    B_avg = []
    for i in np.linspace(1,K,K, dtype = int):
        for j in np.linspace(1,K,K, dtype = int):
            add = np.mean(distance.cdist(clust_points[i], clust_points[j], 'euclidean'))
            B_avg.append(add)
    B = np.mean(B_avg)
    A_avg = []
    for i in np.linspace(1,K,K, dtype = int):
        add = np.mean(distance.cdist(clust_points[i],clust_points[i], 'euclidean'))
        A_avg.append(add)
    A = np.mean(A_avg)
    SC = (B-A)/max(A,B)
    #print(SC)
    
    return WC_SSD, SC


# In[8]:


WC_SSD_single = []
WC_SSD_complete = []
WC_SSD_average = []

SC_single = []
SC_complete = []
SC_average = []

k = [2,4,8,16,32]

for item in k:
    ssd_1, sc_1 = hierarchial(Z_single, item)
    WC_SSD_single.append(ssd_1)
    SC_single.append(sc_1)
    ssd_2, sc_2 = hierarchial(Z_complete, item)
    WC_SSD_complete.append(ssd_2)
    SC_complete.append(sc_2)
    ssd_3, sc_3 = hierarchial(Z_average, item)
    WC_SSD_average.append(ssd_3)
    SC_average.append(sc_3)


# In[9]:


plt.figure(figsize = (20,7))
WC_1, = plt.plot( k, WC_SSD_single,ms = 10, mew=4, marker='s')
WC_2, = plt.plot( k, WC_SSD_complete,ms = 10, mew=4, marker='s')
WC_3, = plt.plot( k, WC_SSD_average,ms = 10, mew=4, marker='s')

plt.xlabel("Number of CLusters")
plt.ylabel("Sum of squares")
plt.title("Sum of Squares vs Number of Clusters")
plt.legend((WC_1, WC_2, WC_3), ('Single Linkage', 'Complete Linkage', 'Average Linkage'))
plt.show()


# In[10]:


plt.figure(figsize = (20,7))
SC_1, = plt.plot( k, SC_single,ms = 10, mew=4, marker='s')
SC_2, = plt.plot( k, SC_complete,ms = 10, mew=4, marker='s')
SC_3, = plt.plot( k, SC_average,ms = 10, mew=4, marker='s')

plt.xlabel("Number of CLusters")
plt.ylabel("Silhouette Coeffecient")
plt.title("Silhouette Coeffecients vs Number of Clusters")
plt.legend((SC_1, SC_2, SC_3), ('Single Linkage', 'Complete Linkage', 'Average Linkage'))
plt.show()


# In[ ]:




