#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from pprint import pprint
from scipy.stats import mode


# In[2]:


trainingSet = pd.read_csv('trainingSet.csv', index_col = 0)
testingSet = pd.read_csv('testSet.csv', index_col = 0)


# In[3]:



def decisionTree(trainingSet,testingSet):
    
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
    
    
    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = 8):
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
        
            yes_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
            no_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        
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
    
    tree = decision_tree_algorithm(train_df, max_depth = 8, min_samples=50)
    
    Training_Accuracy = calc_accuracy(train_df, tree)
    Testing_Accuracy = calc_accuracy(test_df, tree)
    
    
    print("Training Accuracy: " + str(Training_Accuracy))
    print("Testing Accuracy: " + str(Testing_Accuracy))
    
    return Training_Accuracy, Testing_Accuracy


# In[4]:



def bagging(trainingSet,testingSet):
    
    stopping_criteria = 30
    
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


    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = 8):
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

            yes_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
            no_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)

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

    def decision_frame(df, tree):
        classification = df.apply(label, axis = 1, args = (tree,))
        classification_correct = classification == df.decision
        return classification_correct

    classification_df_train = pd.DataFrame()
    classification_df_test = pd.DataFrame()
    
    for i in range(stopping_criteria):
        train_bootstrap = train_df.sample( frac=1, replace=True, weights=None, axis=0)
        tree = decision_tree_algorithm(train_bootstrap, max_depth = 8, min_samples=50)
        result_label_train = decision_frame(train_df, tree)
        result_label_test = decision_frame(test_df, tree)
        classification_df_train[i] = result_label_train
        classification_df_test[i] = result_label_test
    
    class_label_train = mode(classification_df_train.values, axis = -1)[0]
    class_label_test = mode(classification_df_test.values, axis = -1)[0]
    
    Training_Accuracy = class_label_train.mean()
    Testing_Accuracy = class_label_test.mean()
     
    print("Training Accuracy: " + str(Training_Accuracy))
    print("Testing Accuracy: " + str(Testing_Accuracy))

    return Training_Accuracy, Testing_Accuracy

    


# In[5]:



def randomForests(trainingSet,testingSet):
    
    train_df = trainingSet
    test_df = testingSet
    data = train_df.values
    random_subspace = int(round(np.sqrt(len(train_df.columns))))
    n_trees = 30
    max_depth = 8
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
    
    
    def decision_tree_algorithm(df, counter = 0, min_samples = 50, max_depth = 8, random_subspace = random_subspace):
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
    
    def random_forest_algorithm(train_df, n_trees, random_subspace, max_depth):
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
     
    print("Training Accuracy: " + str(Training_Accuracy))
    print("Testing Accuracy: " + str(Testing_Accuracy))

    return Training_Accuracy, Testing_Accuracy
        
        


# In[6]:


def trees(trainingSet, testingSet, modelIdx):
    if modelIdx == 1:
        decisionTree(trainingSet,testingSet)
    elif modelIdx == 2:
        bagging(trainingSet, testingSet)
    elif modelIdx == 3:
        randomForests(trainingSet, testingSet)


# In[8]:


trees(trainingSet, testingSet, 1)


# In[ ]:




