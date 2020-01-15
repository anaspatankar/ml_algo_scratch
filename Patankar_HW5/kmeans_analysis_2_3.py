#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
df_2 = df[(df['label']==2)| (df['label']==4) | (df['label']==6) | (df['label']==7)].reset_index(drop= True)
df_3 = df[(df['label']==6) | (df['label']==7)].reset_index(drop= True)


# In[3]:


def dataset1(K, set_seed):
    df = pd.read_csv("digits-embedding.csv", names = ['index', 'label', 'dim1', 'dim2']).drop('index', axis = 1)
    vertex = list(zip(df['dim1'], df['dim2']))
    np.random.seed(set_seed)
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
    


# In[4]:


def dataset2(K, set_seed):
    df_2 = df[(df['label']==2)| (df['label']==4) | (df['label']==6) | (df['label']==7)].reset_index(drop= True)
    vertex = list(zip(df_2['dim1'], df_2['dim2']))
    np.random.seed(set_seed)
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


# In[5]:


def dataset3(K, set_seed):
    df_3 = df[(df['label']==6) | (df['label']==7)].reset_index(drop= True)
    vertex = list(zip(df_3['dim1'], df_3['dim2']))
    np.random.seed(set_seed)
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


# In[6]:


WC_SSD_1_avg = []
WC_SSD_2_avg = []
WC_SSD_3_avg = []
WC_SSD_1_StandDev = []
WC_SSD_2_StandDev = []
WC_SSD_3_StandDev = []

SC_1_avg = []
SC_2_avg = []
SC_3_avg = []
SC_1_StandDev = []
SC_2_StandDev = []
SC_3_StandDev = []

k = [2,4,8,16,32]

for item in k:
    print(item)
    WC_SSD_1_perseed = []
    WC_SSD_2_perseed = []
    WC_SSD_3_perseed = []

    SC_1_perseed = []
    SC_2_perseed = []
    SC_3_perseed = []

    for seed in np.arange(4,100,10):
        ssd_1, sc_1 = dataset1(item, seed)
        WC_SSD_1_perseed.append(ssd_1)
        SC_1_perseed.append(sc_1)
        ssd_2, sc_2 = dataset2(item, seed)
        WC_SSD_2_perseed.append(ssd_2)
        SC_2_perseed.append(sc_2)
        ssd_3, sc_3 = dataset3(item, seed)
        WC_SSD_3_perseed.append(ssd_3)
        SC_3_perseed.append(sc_3)
    
    
    print("\n")
    #For WC_SSD
    WC_1_perseed = np.mean(WC_SSD_1_perseed)
    WC_SSD_1_avg.append(WC_1_perseed)
    
    

    SD_1_perseed = (np.std(WC_SSD_1_perseed))
    WC_SSD_1_StandDev.append(SD_1_perseed)
    
    

    WC_2_perseed = np.mean(WC_SSD_2_perseed)
    WC_SSD_2_avg.append(WC_2_perseed)
    
    
        
    SD_2_perseed= (np.std(WC_SSD_2_perseed))
    WC_SSD_2_StandDev.append(SD_2_perseed)
    
    
    
    WC_3_perseed = np.mean(WC_SSD_3_perseed)
    WC_SSD_3_avg.append(WC_3_perseed)
    
    
    
    SD_3_perseed = (np.std(WC_SSD_3_perseed))
    WC_SSD_3_StandDev.append(SD_3_perseed) 
    
    
    
    
    
    #For SC
    SCo_1_perseed = np.mean(SC_1_perseed)
    SC_1_avg.append(SCo_1_perseed)
    
    
    

    SD_1_perseed = (np.std(SC_1_perseed))
    SC_1_StandDev.append(SD_1_perseed)
    
    

    SCo_2_perseed = np.mean(SC_2_perseed)
    SC_2_avg.append(SCo_2_perseed)
    
    
        
    SD_2_perseed= (np.std(SC_2_perseed))
    SC_2_StandDev.append(SD_2_perseed)
    
    
    
    SCo_3_perseed = np.mean(SC_3_perseed)
    SC_3_avg.append(SCo_3_perseed)
    
    
    
    SD_3_perseed = (np.std(SC_3_perseed))
    SC_3_StandDev.append(SD_3_perseed) 
    
    
    
    
print(" For k equals to [2,4,8,16,32], the average and SD for WC_SSD, for the averaged value accross 10 seeds are:")   
   
print("Average WC SSD for Dataset 1 equals to " + str(WC_SSD_1_avg))
print("Standard Deviation for WC SSD  Dataset 1 equals to " + str(WC_SSD_1_StandDev))
print("Average WC SSD for Dataset 2 equals to " + str(WC_SSD_2_avg))
print("Standard Deviation for Dataset 2 equals to " + str(WC_SSD_2_StandDev))
print("Average WC SSD for Dataset 3 equals to " + str(WC_SSD_3_avg))
print("Standard Deviation for Dataset 3 equals to " + str(WC_SSD_3_StandDev))
print("\n")

print(" For k equals to [2,4,8,16,32], the average and SD for SC, for the averaged value accross 10 seeds are:")

print("Average SC for Dataset 1 equals to " + str(SC_1_avg))
print("Standard Deviation for Dataset 1 equals to " + str(SC_1_StandDev))
print("Average SC for Dataset 2 equals to " + str(SC_2_avg))
print("Standard Deviation for Dataset 2 equals to " + str(SC_2_StandDev))
print("Average SC for Dataset 3 equals to " + str(SC_3_avg))
print("Standard Deviation for Dataset 3 equals to " + str(SC_3_StandDev))


# WC_SSD

# In[51]:


plt.figure(figsize = (20,8))
WC_1 = plt.errorbar( k, WC_SSD_1_avg, yerr = WC_SSD_1_StandDev, fmt = '-o', ms = 5, mew=4, marker='s')
WC_2 = plt.errorbar( k, WC_SSD_2_avg, yerr = WC_SSD_2_StandDev , fmt = '-o', ms = 5,mew = 4, marker = 'd')
WC_3 = plt.errorbar( k, WC_SSD_3_avg, yerr = WC_SSD_3_StandDev , fmt = '-o', ms = 5 , mew=4, ecolor = "black")
plt.xlabel("K")
plt.ylabel("WC_SSD")
plt.title("WC_SSD vc Number of Clusters for Dataset 1")
#plt.legend((LR, SVM, NBC), ('LR', 'SVM', 'NBC') )
plt.legend((WC_1, WC_2, WC_3), ('Dataset1', 'Dataset2', 'Dataset3'), bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.1)
plt.show()


# SC

# In[52]:


plt.figure(figsize = (20,8))
#plt.gca().set_color_cycle(['red', 'green'])
SC_1 = plt.errorbar( k, SC_1_avg, yerr = SC_1_StandDev, fmt = '-o', ms = 5, mew=4, marker='s')
SC_2 = plt.errorbar( k, SC_2_avg, yerr = SC_2_StandDev , fmt = '-o', ms = 5,mew = 4, marker = 'd')
SC_3 = plt.errorbar( k, SC_3_avg, yerr = SC_3_StandDev , fmt = '-o', ms = 5 , mew=4, ecolor = "black")
plt.xlabel("K")
plt.ylabel("SC")
plt.title("SC vc Number of Clusters for Dataset1")
#plt.legend((LR, SVM, NBC), ('LR', 'SVM', 'NBC') )
plt.legend((SC_1, SC_2, SC_3), ('Dataset1', 'Dataset2', 'Dataset3'), bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.1)
plt.show()


# In[ ]:




