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


# In[6]:


df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
df_2 = df[(df['label']==2)| (df['label']==4) | (df['label']==6) | (df['label']==7)].reset_index(drop= True)
df_3 = df[(df['label']==6) | (df['label']==7)].reset_index(drop= True)


# #Dateset 1

# In[7]:


def dataset1(K):
    df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
    vertex = list(zip(df['dim1'], df['dim2']))
    centroid_index = np.random.randint(0,len(df), size = K)
    centroid = [vertex[x] for x in centroid_index]

    for iteration in range(50):
        #print(iteration)
        centroid_old = centroid
        cluster = []
        array_1 = (np.square(distance.cdist(vertex, centroid)))
        cluster = [np.argmin(array_1[i]) for i in range(len(array_1))]
        cluster = np.array(cluster)
        def index_cluster(cluster_number):
            return (np.where(cluster == cluster_number)[0])
        clust_points = {}
        for num in range(K):
            clust_points[num] = [vertex[i] for i in index_cluster(num)]
        centroid = [(sum((list(zip(*clust_points[num]))[0]))/len(clust_points[num]), sum((list(zip(*clust_points[num]))[1]))/len(clust_points[num])) for num in range(K)]

        if centroid == centroid_old:
            break

    list_ss = []
    for clust_index in range(K):
        for i in range(len(clust_points[clust_index])):
            list_ss.append(np.square(distance.euclidean(centroid[clust_index],clust_points[clust_index][i])))
    WC_SSD = sum(list_ss)

    cluster = []
    for i in range(K):
        cluster.extend(list(np.repeat(i,len(clust_points[i]))))
    dim_1 = []
    dim_2 = []
    for i in range(K):
        x = list(zip(*clust_points[i]))[0]
        y = list(zip(*clust_points[i]))[1]
        dim_1.extend(x)
        dim_2.extend(y)

    dt = pd.DataFrame()
    dt['dim_1'] = dim_1
    dt['dim_2'] = dim_2
    dt['cluster'] = cluster

    dt = dt.sort_values(by = ['dim_1', 'dim_2', 'cluster']).reset_index(drop = True)
    df = df.sort_values(by = ['dim1', 'dim2', 'label']).reset_index(drop = True)

    result = pd.merge(df,dt,on = df.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)

    B_avg = []
    for i in range(K):
        for j in range(K):
            add = np.mean(distance.cdist(clust_points[i], clust_points[j], 'euclidean'))
            B_avg.append(add)
    B = np.mean(B_avg)
    A_avg = []
    for i in range(K):
        add = np.mean(distance.cdist(clust_points[i],clust_points[i], 'euclidean'))
        A_avg.append(add)
    A = np.mean(A_avg)
    SC = (B-A)/max(A,B)

    return WC_SSD, SC
    


# #Dataset 2

# In[8]:


def dataset2(K):
    df_2 = df[(df['label']==2)| (df['label']==4) | (df['label']==6) | (df['label']==7)].reset_index(drop= True)
    vertex = list(zip(df_2['dim1'], df_2['dim2']))
    centroid_index = np.random.randint(0,len(df_2), size = K)
    centroid = [vertex[x] for x in centroid_index]

    for iteration in range(50):
        #print(iteration)
        centroid_old = centroid
        cluster = []
        array_1 = (np.square(distance.cdist(vertex, centroid)))
        cluster = [np.argmin(array_1[i]) for i in range(len(array_1))]
        cluster = np.array(cluster)
        def index_cluster(cluster_number):
            return (np.where(cluster == cluster_number)[0])
        clust_points = {}
        for num in range(K):
            clust_points[num] = [vertex[i] for i in index_cluster(num)]
        centroid = [(sum((list(zip(*clust_points[num]))[0]))/len(clust_points[num]), sum((list(zip(*clust_points[num]))[1]))/len(clust_points[num])) for num in range(K)]

        if centroid == centroid_old:
            break

    list_ss = []
    for clust_index in range(K):
        for i in range(len(clust_points[clust_index])):
            list_ss.append(np.square(distance.euclidean(centroid[clust_index],clust_points[clust_index][i])))
    WC_SSD = sum(list_ss)

    cluster = []
    for i in range(K):
        cluster.extend(list(np.repeat(i,len(clust_points[i]))))
    dim_1 = []
    dim_2 = []
    for i in range(K):
        x = list(zip(*clust_points[i]))[0]
        y = list(zip(*clust_points[i]))[1]
        dim_1.extend(x)
        dim_2.extend(y)

    dt = pd.DataFrame()
    dt['dim_1'] = dim_1
    dt['dim_2'] = dim_2
    dt['cluster'] = cluster

    dt = dt.sort_values(by = ['dim_1', 'dim_2', 'cluster']).reset_index(drop = True)
    df_2 = df_2.sort_values(by = ['dim1', 'dim2', 'label']).reset_index(drop = True)

    result = pd.merge(df_2,dt,on = df_2.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)

    B_avg = []
    for i in range(K):
        for j in range(K):
            add = np.mean(distance.cdist(clust_points[i], clust_points[j], 'euclidean'))
            B_avg.append(add)
    B = np.mean(B_avg)
    A_avg = []
    for i in range(K):
        add = np.mean(distance.cdist(clust_points[i],clust_points[i], 'euclidean'))
        A_avg.append(add)
    A = np.mean(A_avg)
    SC = (B-A)/max(A,B)

    return WC_SSD, SC


# #Dataset 3

# In[9]:


def dataset3(K):
    df_3 = df[(df['label']==6) | (df['label']==7)].reset_index(drop= True)
    vertex = list(zip(df_3['dim1'], df_3['dim2']))
    centroid_index = np.random.randint(0,len(df_3), size = K)
    centroid = [vertex[x] for x in centroid_index]

    for iteration in range(50):
        #print(iteration)
        centroid_old = centroid
        cluster = []
        array_1 = (np.square(distance.cdist(vertex, centroid)))
        cluster = [np.argmin(array_1[i]) for i in range(len(array_1))]
        cluster = np.array(cluster)
        def index_cluster(cluster_number):
            return (np.where(cluster == cluster_number)[0])
        clust_points = {}
        for num in range(K):
            clust_points[num] = [vertex[i] for i in index_cluster(num)]
        centroid = [(sum((list(zip(*clust_points[num]))[0]))/len(clust_points[num]), sum((list(zip(*clust_points[num]))[1]))/len(clust_points[num])) for num in range(K)]

        if centroid == centroid_old:
            break

    list_ss = []
    for clust_index in range(K):
        for i in range(len(clust_points[clust_index])):
            list_ss.append(np.square(distance.euclidean(centroid[clust_index],clust_points[clust_index][i])))
    WC_SSD = sum(list_ss)

    cluster = []
    for i in range(K):
        cluster.extend(list(np.repeat(i,len(clust_points[i]))))
    dim_1 = []
    dim_2 = []
    for i in range(K):
        x = list(zip(*clust_points[i]))[0]
        y = list(zip(*clust_points[i]))[1]
        dim_1.extend(x)
        dim_2.extend(y)

    dt = pd.DataFrame()
    dt['dim_1'] = dim_1
    dt['dim_2'] = dim_2
    dt['cluster'] = cluster

    dt = dt.sort_values(by = ['dim_1', 'dim_2', 'cluster']).reset_index(drop = True)
    df_3 = df_3.sort_values(by = ['dim1', 'dim2', 'label']).reset_index(drop = True)

    result = pd.merge(df_3,dt,on = df_3.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)

    B_avg = []
    for i in range(K):
        for j in range(K):
            add = np.mean(distance.cdist(clust_points[i], clust_points[j], 'euclidean'))
            B_avg.append(add)
    B = np.mean(B_avg)
    A_avg = []
    for i in range(K):
        add = np.mean(distance.cdist(clust_points[i],clust_points[i], 'euclidean'))
        A_avg.append(add)
    A = np.mean(A_avg)
    SC = (B-A)/max(A,B)

    return WC_SSD, SC


# In[10]:


WC_SSD_1 = []
WC_SSD_2 = []
WC_SSD_3 = []

SC_1 = []
SC_2 = []
SC_3 = []

k = [2,4,8,16,32]

for item in k:
    ssd_1, sc_1 = dataset1(item)
    WC_SSD_1.append(ssd_1)
    SC_1.append(sc_1)
    ssd_2, sc_2 = dataset2(item)
    WC_SSD_2.append(ssd_2)
    SC_2.append(sc_2)
    ssd_3, sc_3 = dataset3(item)
    WC_SSD_3.append(ssd_3)
    SC_3.append(sc_3)


# In[11]:


plt.figure(figsize = (20,7))
WC_1, = plt.plot( k, WC_SSD_1,ms = 10, mew=4, marker='s')
WC_2, = plt.plot( k, WC_SSD_2,ms = 10, mew=4, marker='s')
WC_3, = plt.plot( k, WC_SSD_3,ms = 10, mew=4, marker='s')

plt.xlabel("Number of CLusters")
plt.ylabel("Sum of squares")
plt.title("Sum of Squares vs Number of Clusters")
plt.legend((WC_1, WC_2, WC_3), ('Dataset_1', 'Dataset_2', 'Dataset_3'))
plt.show()


# In[60]:


plt.figure(figsize = (20,7))
SC_1, = plt.plot( k, SC_1,ms = 10, mew=4, marker='s')
SC_2, = plt.plot( k, SC_2,ms = 10, mew=4, marker='s')
SC_3, = plt.plot( k, SC_3,ms = 10, mew=4, marker='s')

plt.xlabel("Number of CLusters")
plt.ylabel("Silhouette Coeffecient")
plt.title("Silhouette Coeffecients vs Number of Clusters")
plt.legend((SC_1, SC_2, SC_3), ('Dataset_1', 'Dataset_2', 'Dataset_3'))
plt.show()


# In[ ]:




