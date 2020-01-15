#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
np.random.seed(0)
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
df_2 = df[(df['label']==2)| (df['label']==4) | (df['label']==6) | (df['label']==7)].reset_index(drop= True)
df_3 = df[(df['label']==6) | (df['label']==7)].reset_index(drop= True)


# In[93]:


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
        #print(cluster)
        def index_cluster(cluster_number):
            return (np.where(cluster == cluster_number)[0])
        clust_points = {}

        for num in range(K):
            clust_points[num] = [vertex[i] for i in index_cluster(num)]
        #print(clust_points)
        centroid = [(sum((list(zip(*clust_points[num]))[0]))/len(clust_points[num]), sum((list(zip(*clust_points[num]))[1]))/len(clust_points[num])) for num in range(K)]

        if centroid == centroid_old:
            break

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

    df = pd.merge(df,dt,on = df.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)
    
    
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
    for i in range(K):
        X = np.array([prob_cg[i][cluster]*(np.log(prob_cg[i][cluster]/(prob_class[i] * prob_clusters[i]))) for cluster in range(K)])
        X = X[~np.isnan(X)]
        X[X<0] = 0
        Y = np.sum(X)
        list1.append(Y)
    
    NMI = np.sum(list1)/(class_entropy+cluster_entropy)
    
    return df, NMI


# In[94]:


df_1, NMI_Dataset1 = dataset1(8)


# In[115]:


print("NMI Dataset 1: " + str(NMI_Dataset1))


# In[96]:


points = np.random.randint(0, len(df_1), size = 1000)
sns.FacetGrid(df_1.iloc[points, :], hue = 'cluster', size =6).map(plt.scatter, 'dim1', 'dim2')
plt.legend(bbox_to_anchor=(1.1, 1), borderaxespad=0.1)
plt.show()


# In[97]:


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

    cluster = []
    for i in range(K):
        cluster.extend(list(np.repeat(i,len(clust_points[i]))))
    #print(cluster)
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

    df_2 = pd.merge(df_2,dt,on = df_2.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)
    
    
    class_count = df_2.groupby('label').count()['cluster']
    cluster_count = df_2.groupby('cluster').count()['label']
    
    class_entropy = sum(-np.multiply(np.array(class_count/len(df_2)), np.log(np.array(class_count/len(df_2)))))
    cluster_entropy = sum(-np.multiply(np.array(cluster_count/len(df_2)), np.log(np.array(cluster_count/len(df_2)))))
    
    
    N = len(df_2)
    class_num = sorted(list(df_2['label'].unique()))
    cluster_num = sorted(list(df_2['cluster'].unique()))
    prob_class = {i:(len(df_2[df_2['label']==i])/N) for i in class_num}
    
    prob_class[0] = prob_class.pop(2)
    prob_class[1] = prob_class.pop(4)
    prob_class[2] = prob_class.pop(6)
    prob_class[3] = prob_class.pop(7)
    
    prob_clusters = {i:(len(df_2[df_2['cluster']==i])/N) for i in cluster_num}
    prob_cg = {i: {j:(len(df_2[(df_2['label']==i) & (df_2['cluster']==j)])/N) for j in cluster_num} for i in class_num}
    
    prob_cg[0] = prob_cg.pop(2)
    prob_cg[1] = prob_cg.pop(4)
    prob_cg[2] = prob_cg.pop(6)
    prob_cg[3] = prob_cg.pop(7)
    
    list1 = []
    for i in range(3):
        X = np.array([prob_cg[i][cluster]*(np.log(prob_cg[i][cluster]/(prob_class[i] * prob_clusters[i]))) for cluster in range(K)])
        X = X[~np.isnan(X)]
        X[X<0] = 0
        Y = np.sum(X)
        list1.append(Y)
    
    NMI = np.sum(list1)/(class_entropy+cluster_entropy)
    
    return df_2, NMI


# In[98]:


df_2, NMI_Dataset2 = dataset2(4)


# In[116]:


print("NMI Dataset 2: " + str(NMI_Dataset2))


# In[103]:


points = np.random.randint(0, len(df_2), size = 1000)
sns.FacetGrid(df_2.iloc[points, :], hue = 'cluster', size =6).map(plt.scatter, 'dim1', 'dim2')
plt.legend(bbox_to_anchor=(1.1, 1), borderaxespad=0.1)
plt.show()


# In[106]:


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

    cluster = []
    for i in range(K):
        cluster.extend(list(np.repeat(i,len(clust_points[i]))))
    #print(cluster)
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

    df_3 = pd.merge(df_3,dt,on = df_3.index ).drop(['key_0','dim_1', 'dim_2'], axis = 1)
    
    
    class_count = df_3.groupby('label').count()['cluster']
    cluster_count = df_3.groupby('cluster').count()['label']
    
    class_entropy = sum(-np.multiply(np.array(class_count/len(df_3)), np.log(np.array(class_count/len(df_3)))))
    cluster_entropy = sum(-np.multiply(np.array(cluster_count/len(df_3)), np.log(np.array(cluster_count/len(df_3)))))
    
    
    N = len(df_3)
    class_num = sorted(list(df_3['label'].unique()))
    cluster_num = sorted(list(df_3['cluster'].unique()))
    prob_class = {i:(len(df_3[df_3['label']==i])/N) for i in class_num}
    
    prob_class[0] = prob_class.pop(6)
    prob_class[1] = prob_class.pop(7)
    
    prob_clusters = {i:(len(df_3[df_3['cluster']==i])/N) for i in cluster_num}
    prob_cg = {i: {j:(len(df_3[(df_3['label']==i) & (df_3['cluster']==j)])/N) for j in cluster_num} for i in class_num}
    
    prob_cg[0] = prob_cg.pop(6)
    prob_cg[1] = prob_cg.pop(7)
    
    
    list1 = []
    for i in range(2):
        X = np.array([prob_cg[i][cluster]*(np.log(prob_cg[i][cluster]/(prob_class[i] * prob_clusters[i]))) for cluster in range(K)])
        X = X[~np.isnan(X)]
        X[X<0] = 0
        Y = np.sum(X)
        list1.append(Y)
    
    NMI = np.sum(list1)/(class_entropy+cluster_entropy)
    
    return df_3, NMI


# In[107]:


df_3, NMI_Dataset3 = dataset3(2)


# In[117]:


print("NMI Dataset 3: " + str(NMI_Dataset3))


# In[108]:


points = np.random.randint(0, len(df_3), size = 1000)
sns.FacetGrid(df_3.iloc[points, :], hue = 'cluster', size =6).map(plt.scatter, 'dim1', 'dim2')
plt.legend(bbox_to_anchor=(1.1, 1), borderaxespad=0.1)
plt.show()


# In[110]:


NMI = [NMI_Dataset1, NMI_Dataset2, NMI_Dataset3]
Dataset = [1,2,3]
sns.barplot(Dataset, NMI)


# In[ ]:




