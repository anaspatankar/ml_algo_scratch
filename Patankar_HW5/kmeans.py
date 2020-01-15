#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
np.random.seed(0)
from scipy.spatial import distance


# In[2]:


df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)


# In[19]:


def kmeans(dataset, K):
    
    df = dataset
    points = np.arange(0, len(df))
    vertex = list(zip(df['dim1'], df['dim2']))
    centroid_index = np.random.randint(0,len(df), size = 10)
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
    print("WC-SSD: "+ str(WC_SSD))
    
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

    print("SC: " + str(SC))
    
    
    def NMI(dataset):
    
        df = result
        
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

        return NMI

        
    
    print("NMI: " + str(NMI(df)))
    


# In[20]:


kmeans(df, 10)


# In[ ]:




