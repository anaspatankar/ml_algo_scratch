#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import scipy
from pprint import pprint
from scipy.stats import mode


# In[2]:


def decisionTree(trainingSet,testingSet, max_depth):
    
    max_depth = max_depth
    train_df = trainingSet
    test_df = testingSet
    data = train_df.values
    
    def unique_label_func(data):
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        else:
            return False

    
    def classification_func(data):
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]   
        return classification

    
    
    def potential_splits_cols(data):
        potential_splits = {}
        _, n_columns = data.shape
        for column_index in range(n_columns-1):
            potential_splits[column_index] = []
            values = data[:,column_index]
            potential_split = np.unique(values)
            potential_splits[column_index].append(potential_split)
    
        return potential_splits
    
    
    def split_data(data, split_column, split_value):
        split_column_values = data[:,split_column]
        data_below = data[split_column_values == 0]
        data_above = data[split_column_values == 1]
        return data_below, data_above


    def gini(data):
    
        label_column = data[:, -1]
        _,counts = np.unique(label_column, return_counts=True)
    
        probabilities = counts / counts.sum()
        gini = 1 - sum(np.square(probabilities))
    
        return gini

    def cumulative_gini(data_below, data_above):
    
        n_data_points = len(data_below) + len(data_above)
        p_data_below = len(data_below)/n_data_points
        p_data_above = len(data_above)/n_data_points
        overall_gini = (p_data_below * gini(data_below) + 
                p_data_above * gini(data_above))
        return overall_gini
    
    def determine_best_spit(data, potential_splits):
        overall_gini = 100
        for column_index in potential_splits:
            for value in potential_splits[column_index][0]:
            #print(value
                data_below, data_above = split_data(data, split_column = column_index, split_value = value)
                current_overall_gini = cumulative_gini(data_below, data_above)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value
        return best_split_column, best_split_value
    
    
    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = max_depth):
        if counter == 0:
            global column_headers
            column_headers = train_df.columns
            data = df.values
        else:
            data = df
        if (unique_label_func(data)) or (len(data)<min_samples) or (counter == max_depth):
            classification = classification_func(data)
            return classification
        else:
            counter+=1
            potential_splits = potential_splits_cols(data)
            split_column, split_value = determine_best_spit(data, potential_splits)
            data_below, data_above = split_data(data, split_column, split_value)
        
            feature_name = column_headers[split_column]
            question = "{} == {}".format(feature_name, split_value)
            sub_tree = {question: []}
        
            yes_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth = max_depth)
            no_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth = max_depth)
        
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
        
            return sub_tree

    def label(example, tree):
        question = list(tree.keys())[0]
        feature_name, comparision_operator, value = question.split()

        if example[feature_name] == int(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return label(example, answer)
    
    def calc_accuracy(df, tree):
    
        classification = df.apply(label, axis = 1, args = (tree,))
        classification_correct = classification == df.decision
        accuracy = classification_correct.mean()
        return accuracy
    
    tree = decision_tree_algorithm(train_df, max_depth = max_depth, min_samples=50)
    
    Training_Accuracy = calc_accuracy(train_df, tree)
    Testing_Accuracy = calc_accuracy(test_df, tree)
    
    
    #print("Training Accuracy: " + str(Training_Accuracy))
    #print("Testing Accuracy: " + str(Testing_Accuracy))
    
    return round(Testing_Accuracy,2)


# In[3]:



def bagging(trainingSet,testingSet, max_depth, n_trees):
    
    train_df = trainingSet
    test_df = testingSet
    data = train_df.values
    n_trees = n_trees
    max_depth = max_depth
    min_samples = 50
    
    def unique_label_func(data):
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        else:
            return False


    def classification_func(data):
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]   
        return classification



    def potential_splits_cols(data):
        potential_splits = {}
        _, n_columns = data.shape
        for column_index in range(n_columns-1):
            potential_splits[column_index] = []
            values = data[:,column_index]
            potential_split = np.unique(values)
            potential_splits[column_index].append(potential_split)

        return potential_splits


    def split_data(data, split_column, split_value):
        split_column_values = data[:,split_column]
        data_below = data[split_column_values == 0]
        data_above = data[split_column_values == 1]
        return data_below, data_above


    def gini(data):

        label_column = data[:, -1]
        _,counts = np.unique(label_column, return_counts=True)

        probabilities = counts / counts.sum()
        gini = 1 - sum(np.square(probabilities))

        return gini

    def cumulative_gini(data_below, data_above):

        n_data_points = len(data_below) + len(data_above)
        p_data_below = len(data_below)/n_data_points
        p_data_above = len(data_above)/n_data_points
        overall_gini = (p_data_below * gini(data_below) + 
                p_data_above * gini(data_above))
        return overall_gini

    def determine_best_spit(data, potential_splits):
        overall_gini = 100
        for column_index in potential_splits:
            for value in potential_splits[column_index][0]:
            #print(value
                data_below, data_above = split_data(data, split_column = column_index, split_value = value)
                current_overall_gini = cumulative_gini(data_below, data_above)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value
        return best_split_column, best_split_value


    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = max_depth):
        if counter == 0:
            global column_headers
            column_headers = train_df.columns
            data = df.values
        else:
            data = df
        if (unique_label_func(data)) or (len(data)<min_samples) or (counter == max_depth):
            classification = classification_func(data)
            return classification
        else:
            counter+=1
            potential_splits = potential_splits_cols(data)
            split_column, split_value = determine_best_spit(data, potential_splits)
            data_below, data_above = split_data(data, split_column, split_value)

            feature_name = column_headers[split_column]
            question = "{} == {}".format(feature_name, split_value)
            sub_tree = {question: []}

            yes_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth = max_depth)
            no_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth = max_depth)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree
        
    def bootstrapping(train_df):
        bootstrap_indices = np.random.randint(low = 0, high = len(train_df), size = len(train_df))
        df_bootstrapped = train_df.iloc[bootstrap_indices]
        return df_bootstrapped

    
    
    
    def label(example, tree):
        question = list(tree.keys())[0]
        feature_name, comparision_operator, value = question.split()

        if example[feature_name] == int(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return label(example, answer)

    def decision_frame(df, tree):
        classification = df.apply(label, axis = 1, args = (tree,))
        classification_correct = classification == df.decision
        return classification_correct
    
    def bagged_tree_algorithm(train_df, n_trees = n_trees, max_depth = max_depth):
        bagged = []
        for i in range(n_trees):
            df_boostrapped = bootstrapping(train_df)
            tree = decision_tree_algorithm(df_boostrapped, min_samples = min_samples, max_depth = max_depth)
            bagged.append(tree)
        return bagged
    
    bagged = bagged_tree_algorithm(train_df, n_trees = n_trees, max_depth = max_depth)
    
    classification_df_train = pd.DataFrame()
    classification_df_test = pd.DataFrame()
    
    for i in range(len(bagged)):
        #print(i)
        result_label_train = decision_frame(train_df, bagged[i])
        result_label_test = decision_frame(test_df, bagged[i])
        classification_df_train[i] = result_label_train
        classification_df_test[i] = result_label_test
    
    class_label_train = mode(classification_df_train.values, axis = -1)[0]
    class_label_test = mode(classification_df_test.values, axis = -1)[0]
    
    Training_Accuracy = class_label_train.mean()
    Testing_Accuracy = class_label_test.mean()
     
    #print("Training Accuracy: " + str(Training_Accuracy))
    #print("Testing Accuracy: " + str(Testing_Accuracy))

    return round(Testing_Accuracy,2)
        

    


# In[4]:



def randomForests(trainingSet,testingSet, max_depth, n_trees):
    
    train_df = trainingSet
    test_df = testingSet
    data = train_df.values
    random_subspace = int(round(np.sqrt(len(train_df.columns))))
    n_trees = n_trees
    max_depth = max_depth
    min_samples = 50
    
    def unique_label_func(data):
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        else:
            return False

    
    def classification_func(data):
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]   
        return classification

    
    
    def potential_splits_cols(data, random_subspace = random_subspace):
        potential_splits = {}
        _, n_columns = data.shape
        column_indices = list(range(n_columns-1))

        if random_subspace:
            column_indices = random.sample(population= column_indices, k = random_subspace)

        for column_index in column_indices:
            potential_splits[column_index] = []
            values = data[:,column_index]
            potential_split = np.unique(values)
            potential_splits[column_index].append(potential_split)

        return potential_splits
    
    def split_data(data, split_column, split_value):
        split_column_values = data[:,split_column]
        data_below = data[split_column_values == 0]
        data_above = data[split_column_values == 1]
        return data_below, data_above


    def gini(data):
    
        label_column = data[:, -1]
        _,counts = np.unique(label_column, return_counts=True)
    
        probabilities = counts / counts.sum()
        gini = 1 - sum(np.square(probabilities))
    
        return gini

    def cumulative_gini(data_below, data_above):
    
        n_data_points = len(data_below) + len(data_above)
        p_data_below = len(data_below)/n_data_points
        p_data_above = len(data_above)/n_data_points
        overall_gini = (p_data_below * gini(data_below) + 
                p_data_above * gini(data_above))
        return overall_gini
    
    def determine_best_spit(data, potential_splits):
        overall_gini = 100
        for column_index in potential_splits:
            for value in potential_splits[column_index][0]:
            #print(value
                data_below, data_above = split_data(data, split_column = column_index, split_value = value)
                current_overall_gini = cumulative_gini(data_below, data_above)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value
        return best_split_column, best_split_value
    
    
    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = max_depth, random_subspace = random_subspace):
        if counter == 0:
            global column_headers
            column_headers = train_df.columns
            data = df.values
        else:
            data = df
        if (unique_label_func(data)) or (len(data)<min_samples) or (counter == max_depth):
            classification = classification_func(data)
            return classification
        else:
            counter+=1
            potential_splits = potential_splits_cols(data)
            split_column, split_value = determine_best_spit(data, potential_splits)
            data_below, data_above = split_data(data, split_column, split_value)
        
            feature_name = column_headers[split_column]
            question = "{} == {}".format(feature_name, split_value)
            sub_tree = {question: []}
        
            yes_answer = decision_tree_algorithm(data_above, counter, min_samples = min_samples, max_depth = max_depth, random_subspace = random_subspace)
            no_answer = decision_tree_algorithm(data_below, counter, min_samples = min_samples, max_depth = max_depth, random_subspace = random_subspace)
        
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
        
            return sub_tree

    def bootstrapping(train_df):
        bootstrap_indices = np.random.randint(low = 0, high = len(train_df), size = len(train_df))
        df_bootstrapped = train_df.iloc[bootstrap_indices]
        return df_bootstrapped

    def label(example, tree):
        question = list(tree.keys())[0]
        feature_name, comparision_operator, value = question.split()

        if example[feature_name] == int(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return label(example, answer)
    
    def decision_frame(df, tree):
        classification = df.apply(label, axis = 1, args = (tree,))
        classification_correct = classification == df.decision
        return classification_correct
    
    def random_forest_algorithm(train_df, random_subspace,n_trees = n_trees, max_depth = max_depth):
        forest = []
        for i in range(n_trees):
            df_boostrapped = bootstrapping(train_df)
            tree = decision_tree_algorithm(df_boostrapped, min_samples = min_samples, max_depth = max_depth, random_subspace = random_subspace)
            forest.append(tree)
        return forest
    
    forest = random_forest_algorithm(train_df, n_trees = n_trees, random_subspace = random_subspace, max_depth = max_depth)
    
    classification_df_train = pd.DataFrame()
    classification_df_test = pd.DataFrame()
    
    for i in range(len(forest)):
        #print(i)
        result_label_train = decision_frame(train_df, forest[i])
        result_label_test = decision_frame(test_df, forest[i])
        classification_df_train[i] = result_label_train
        classification_df_test[i] = result_label_test
    
    class_label_train = mode(classification_df_train.values, axis = -1)[0]
    class_label_test = mode(classification_df_test.values, axis = -1)[0]
    
    Training_Accuracy = class_label_train.mean()
    Testing_Accuracy = class_label_test.mean()
     
    #print("Training Accuracy: " + str(Training_Accuracy))
    #print("Testing Accuracy: " + str(Testing_Accuracy))

    return round(Testing_Accuracy, 2)
        
        


# # Influence of Tree Depth on Classifier Performance

# In[5]:


trainingSet = pd.read_csv('trainingSet.csv', index_col = 0)
trainingSet = trainingSet.sample( frac=1, replace=False, weights=None, random_state=18, axis=0)
trainingSet = trainingSet.sample( frac=0.5, replace=False, weights=None, random_state=32, axis=0)


# In[6]:


limit = np.linspace(0, len(trainingSet), 11)
folds = []
for i in range(len(limit) - 1):
    frow = limit[i]
    lrow = limit[i+1]
    folds.append(list(np.arange(frow,lrow)))


# In[9]:


deciontree_average_depth = []
Bagging_average_depth = []
randomforest_average_depth = []

deciontree_standarderror_depth = []
Bagging_standarderror_depth = []
randomforest_standarderror_depth = []

depth = [3,5,7,9]

for j in depth:
    
    decisiontree = [] 
    baggedtree = [] 
    randomforest = [] 
    
    for i in range(len(folds)):
    
        test_rows = folds[i]
        test_set = trainingSet.iloc[test_rows, :]
        fullset = list(np.arange(0, len(trainingSet)))
        train_rows = [x for x in fullset if x not in folds[i]]
        
        train_set = trainingSet.iloc[train_rows, :]
        
        acc_DT  = decisionTree(train_set, test_set, max_depth = j)
        acc_BA = bagging(train_set, test_set, max_depth = j, n_trees = 8)
        acc_RF  = randomForests(train_set, test_set, max_depth = j, n_trees = 8)
        
        decisiontree.append(acc_DT)
        baggedtree.append(acc_BA)
        randomforest.append(acc_RF)
        
    
    print("Accuracies For single Decision Tree with max depth = " + str(j))
    print(decisiontree)
    
    print("Accuracies For Bagged Decision Tree with max depth = " + str(j))
    print(baggedtree)
    
    print("Accuracies For Random Forest with max depth = " + str(j))
    print(randomforest)
    
    print("\n")
    
    Avg_accuracy_DT = np.mean(decisiontree)
    deciontree_average_depth.append(Avg_accuracy_DT)

    Standard_error_DT = (np.std(decisiontree)/np.sqrt(10))
    deciontree_standarderror_depth.append(Standard_error_DT)

    Avg_accuracy_BA = np.mean(baggedtree)
    Bagging_average_depth .append(Avg_accuracy_BA)
        
    Standard_error_BA = (np.std(baggedtree)/np.sqrt(10))
    Bagging_standarderror_depth.append(Standard_error_BA)
    
    Avg_accuracy_RF = np.mean(randomforest)
    randomforest_average_depth.append(Avg_accuracy_RF)
    
    Standard_error_RF = (np.std(randomforest)/np.sqrt(10))
    randomforest_standarderror_depth.append(Standard_error_RF)

print("\n")
print("Average accuracy accross 10 folds for Decision Tree: " + str(deciontree_average_depth))
print("Average Standard error accross 10 folds for Decision Tree: " + str(deciontree_standarderror_depth))

print("\n")
print("Average accuracy accross 10 folds for Bagged Decision Tree: " + str(Bagging_average_depth))
print("Average Standard error accross 10 folds for Bagged Decision Tree: " + str(Bagging_standarderror_depth))

print("\n")
print("Average accuracy accross 10 folds for Random Forest: " + str(randomforest_average_depth))
print("Average Standard error accross 10 folds for Random Forest: " + str(randomforest_standarderror_depth))


# In[10]:


plt.figure(figsize = (10,8))
DT = plt.errorbar( depth, deciontree_average_depth, yerr = deciontree_standarderror_depth, fmt = '--o', ms = 10, mew=4, marker='s')
BA = plt.errorbar( depth, Bagging_average_depth, yerr = Bagging_standarderror_depth, fmt = '--o', ms = 10 , mew=4, marker = 'd')
RF = plt.errorbar( depth, randomforest_average_depth, yerr = randomforest_standarderror_depth , fmt = '--o', ms = 10 , mew=4)
plt.xlabel("Depth of tree")
plt.ylabel("Accuracy")
plt.title("Mean Accuarcy vc Depth of tree ")
plt.legend((DT, BA, RF), ('Single Decision Tree', 'Bagged Decision Tree', 'Random Forest'), bbox_to_anchor=(0.73, 0.12), loc=2, borderaxespad=0.1)
plt.show()


# Hypothesis Testing: Performance between Random Forest and Decision Tree accross different depths

# depth = 3

# In[18]:


scipy.stats.ttest_ind([0.73, 0.77, 0.73, 0.69, 0.79, 0.75, 0.75, 0.76, 0.73, 0.7], [0.68, 0.67, 0.67, 0.64, 0.7, 0.67, 0.64, 0.73, 0.72, 0.68])


# depth = 5

# In[19]:


scipy.stats.ttest_ind([0.7, 0.75, 0.72, 0.69, 0.75, 0.73, 0.72, 0.78, 0.75, 0.72],[0.7, 0.72, 0.71, 0.67, 0.72, 0.72, 0.65, 0.74, 0.7, 0.67])


# depth = 7

# In[20]:


scipy.stats.ttest_ind([0.69, 0.75, 0.72, 0.67, 0.75, 0.72, 0.71, 0.81, 0.74, 0.69], [0.7, 0.73, 0.71, 0.68, 0.72, 0.73, 0.71, 0.74, 0.69, 0.67])


# depth = 9

# In[21]:


scipy.stats.ttest_ind([0.69, 0.72, 0.71, 0.68, 0.7, 0.7, 0.72, 0.79, 0.73, 0.69],[0.7, 0.66, 0.7, 0.66, 0.72, 0.71, 0.7, 0.75, 0.72, 0.7])


# In[ ]:




