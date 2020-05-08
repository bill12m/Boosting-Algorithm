import subprocess as sp
import pandas as pd
import numpy as np

sp.call('clear', shell = True)

epsilon = np.finfo(float).eps
raw_data = pd.read_csv('Data.csv').set_index('Instance')
data_train = raw_data.iloc[0:10,:]
data_val = raw_data.iloc[10:15,:]
del(raw_data)

def parent_gini(df):
    Class = df.keys()[-1]
    gini = 1
    targets = df[Class].unique()
    
    for target in targets:
        probability = df[Class].value_counts()[target]/len(df[Class])
        gini += -1*np.square(probability)
    return gini

def child_gini(df,attribute):
    Class = df.keys()[-1]
    targets = df[Class].unique()
    variables = df[attribute].unique()
    gini_child = 0
    for variable in variables:
        gini_individual = 1
        for target in targets:
            numerator = len(df[attribute][df[attribute] == variable][df[Class] == target])
            denominator = len(df[attribute][df[attribute] == variable])
            probability_individual = numerator/(denominator + epsilon)
            gini_individual += - np.square(probability_individual)
        probability_child = denominator/len(df)
        gini_child += probability_child * gini_individual
    return gini_child

def parent_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    targets = df[Class].unique()
    for target in targets:
        probability = df[Class].value_counts()[target]/len(df[Class])
        entropy += -probability*np.log2(probability)
    return entropy

def child_entropy(df,attribute):
    Class = df.keys()[-1]
    targets = df[Class].unique()
    variables = df[attribute].unique()
    entropy_child = 0
    for variable in variables:
        entropy_individual = 0
        for target in targets:
            numerator = len(df[attribute][df[attribute] == variable][df[Class] == target])
            denominator = len(df[attribute][df[attribute] == variable])
            probability_individual = numerator/(denominator + epsilon)
            entropy_individual += -probability_individual * np.log2(probability_individual+epsilon)
        probability_child = denominator/len(df)
        entropy_child += probability_child * entropy_individual
    return abs(entropy_child)

def choose_node(df):
    information_gain = []
    for key in df.keys()[:-1]:
        information_gain.append(parent_entropy(df) - child_entropy(df,key))
    return df.keys()[:-1][np.argmax(information_gain)]

def build_subtree(df,node,value):
    return df[df[node] == value].reset_index(drop = True)

def build_decision_tree(df,tree = None):
    #Class = df.keys()[-1]
    node = choose_node(df)
    attribute_values = np.unique(df[node])
    
    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in attribute_values:
        subtree = build_subtree(df,node,value)
        class_values,counts = np.unique(subtree['Class'],return_counts = True)
        
        if len(counts) == 1:
            tree[node][value] = class_values[0]
        else:
            tree[node][value] = build_decision_tree(subtree)
    
    return tree

tree = build_decision_tree(data_train)
from pprint import pprint
pprint(tree, width = 1)

bagging_ensemble = {}
BOOTSTRAP = 10
for boot in range(1,BOOTSTRAP+1):
    new_data_train = data_train.sample(len(data_train), replace = True).reset_index()
    new_data_train = new_data_train.drop(columns = ['Instance'])
    bagging_ensemble[boot] = {}
    tree = build_decision_tree(new_data_train)
    bagging_ensemble[boot] = tree

pprint(bagging_ensemble, width = 1)

def predict(inst,tree):
    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction

final_predictions = []
for index in data_val.index:
    count = 0
    for boot in range(1,BOOTSTRAP+1):
        prediction = predict(data_val.loc[index,:],bagging_ensemble[boot])
        count += prediction
    #print(index, ": ", count)
    if count <= 0:
        final_predictions.append(-1)
    else:
        final_predictions.append(1)
    
for index in data_val.index:
    if data_val.loc[index,'Class'] == final_predictions[index - 11]:
        print(index, ": ", final_predictions[index - 11], "Classified")
    else:
        print(index, ": ", final_predictions[index - 11], "Misclassified")