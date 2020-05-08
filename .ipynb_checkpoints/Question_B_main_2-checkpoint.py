import pandas as pd
import numpy as np
import subprocess as sp
from pprint import pprint

sp.call('clear', shell = True)
#Functions for calculating variables in the program
#Calculate the updated probabilities each epoch
def probability_next(error,probability,classified):
    if classified == False:
        probability_next = probability/(2 * (error + epsilon))
    else:
        probability_next = probability/(2 * (1 - error + epsilon))
    return probability_next
#Calculate the weight for each individual model
def calculate_weight(error):
    if error == 0:
        alpha = 2
        return alpha
    
    alpha = 0.5 * np.log((1 - error + epsilon)/(error + epsilon))
    return alpha
#Calculate the error each epoch
def calculate_error(probabilities):
    error = np.sum(probabilities)
    return error   
#Calculate the entropy
def calculate_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    targets = df[Class].unique()
    for target in targets:
        probability = df[Class].value_counts()[target]/len(df[Class])
        entropy += -1 * probability*np.log2(probability)
    return entropy
##############################################################################
#Below are functions I reused from Project 2. Each of them has been updated in
#some way.
#Finding the optimal splitting point for each level of the decision tree
def choose_node(df,information_gain,splits):
    #Check if all instances have the same class. If they do, take the average
    #as the splitting point.
    uniques = df['Class'].nunique()
    if uniques == 1:
        split = df['x'].mean()
    else:
        for index in df.index:
            #Create a list of splits between the instances
            split = df.loc[index,'x'] + 0.01
            splits.append(split)
            
            #Dataframes to represent the child nodes created by each split
            child_1 = pd.DataFrame(df[df['x'] <= split])
            child_2 = pd.DataFrame(df[df['x'] > split])
            
            #Calculate entropy and save information gain to a list
            weighted_average = (len(child_1)/len(df)) * calculate_entropy(child_1) + (len(child_2)/len(df)) * calculate_entropy(df)
            information_gain.append(1 - weighted_average)
            
        #Choose the splitting condition with the lowest entropy.
        split = splits[np.argmax(np.asarray(information_gain))]
            
    return split
#Create a child node for each level of the decision tree.
def build_subtree(df,split,child):
    if child == 0:
        return df[df['x'] <= split].reset_index(drop = True)
    else:
        return df[df['x'] > split].reset_index(drop = True)
#Main function for building the decision tree
def build_decision_tree(df,information_gain,splits,tree = None):
    #Calculate the optimal splitting condition
    split = choose_node(df,information_gain,splits)
    children = [0,1]
    
    #Create the framework for a new subtree
    if tree is None:
        tree = {}
        tree[split] = {}
    #This checks if the current training set has the same instance for every
    #row. This only happens about 0.1% of the time, but causes problems once
    #the tree gets passed to the predict function.
    if df['x'].nunique() == 1:
        tree[split][children[0]] = df['Class'].max()
        tree[split][children[1]] = df['Class'].max()
        return tree
    
    #Decide what to construct for each of the node's children, a leaf or a
    #subtree
    for child in children:
        subtree = build_subtree(df,split,child)
        class_values,counts = np.unique(subtree['Class'],return_counts = True)
        
        if len(counts) == 1:
            tree[split][child] = class_values[0]
        else:
            tree[split][child ] = build_decision_tree(
                subtree,
                information_gain = [],
                splits = [])
    
    return tree
#Predict and instance's class using the constructed tree
def predict(inst,tree):
    for nodes in tree.keys():   
        #Determine which child node to select
        if inst['x'] <= nodes:
            value = 0
        else:
            value = 1
        
        #Take the value of whatever is on the current child node, whether it's
        #a leaf or subtree
        tree = tree[nodes][value]
        prediction = 0
        
        #Check if you've reached a leaf node or another subtree
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction


#Build training dataframe and calculate original probabilities
data_train = pd.DataFrame(np.array([[0.5,-1],[3.0,-1],[4.5,1],[4.6,1],[4.9,1],
                                    [5.2,-1],[5.3,-1],[5.5,1],[7.0,-1],[9.5,-1]])
                          , columns = ['x','Class'])
data_train['Probabilities'] = np.zeros(shape = (len(data_train)))
for instance in data_train.index:
    data_train.loc[instance,'Probabilities'] += 1/(len(data_train))
print("Original Probabilities\n", data_train['Probabilities'])
#Global variables
epochs = 100
epoch = 0
model_weights = []
epsilon = np.finfo(float).eps

##############################################################################
#Here starts the main part for the program

boosting_ensemble = {}
while epoch < epochs:
    #Sample for new training data with the current set of probabilities
    new_data_train = data_train.iloc[:,0:2].sample(
        n = len(data_train),
        replace = True,
        weights = data_train['Probabilities']).sort_values(
            by = ['x']).reset_index()
    del new_data_train['index']
    
    #Construct decision tree and save it to a dictionary
    boosting_ensemble[epoch + 1] = {}
    tree = build_decision_tree(
        new_data_train,
        information_gain = [],
        splits = [])
    boosting_ensemble[epoch + 1] = tree
    
    #Run the decision tree on the original data set and label each instance
    #as to whether it was classified or not
    data_train['Correctly Classified'] = np.zeros(shape = len(data_train))
    #Run each instance through the prediction function
    for index in data_train.index:
        predicted_class = predict(data_train.loc[index,:],tree)
        if data_train.loc[index,'Class'] == predicted_class:
            data_train.loc[index,'Correctly Classified'] = 1
        else:
            data_train.loc[index,'Correctly Classified'] = 0
    
    #Calculate the error and break the loop if the model's error satisfies
    #the stopping condition
    error = calculate_error(
        data_train['Probabilities'][data_train['Correctly Classified'] == 0])
    if error >= 0.5:
        print("Boosting Round ", epoch + 1, ". Error, ", error)
        break
    
    #Calculate the weight for this model
    model_weights.append(calculate_weight(error))
    
    #Update the probabilities
    for index in data_train.index:
        #Case where the instance was classified
        if data_train.loc[index,'Correctly Classified'] == 1:
            data_train.loc[index,'Probabilities'] = probability_next(
                error = error,
                probability = data_train.loc[index,'Probabilities'],
                classified = True)
        #Case where the instance was misclassified
        else:
            data_train.loc[index,'Probabilities'] = probability_next(
                error = error,
                probability = data_train.loc[index,'Probabilities'],
                classified = False)
    
    #Printing the results
    print("Boosting Round ", epoch +1)
    print("Training Data Set:\n",new_data_train['x'])
    print("\nDecision Tree: ")
    pprint(boosting_ensemble[epoch + 1], width = 1)
    print("\nError: ", error)
    print("Model Weight: ", model_weights[-1])
    print("\nUpdated Probabilities:\n",data_train[['x','Probabilities']], "\n\n")
    epoch += 1

data_val = pd.DataFrame(
    np.arange(1,11,1),
    columns = ['x'])

for tree in boosting_ensemble.keys():
    data_val[tree] = np.zeros(shape = len(data_val))
    for index in data_val.index:
        data_val.loc[index,tree] = model_weights[tree - 1] * predict(
            data_val.loc[index,:],boosting_ensemble[tree])

data_val = data_val.set_index('x')
data_val['Class'] = np.zeros(shape = len(data_val))

for index in data_val.index:
    if data_val.loc[index,:].sum() > 0:
        data_val.loc[index,'Class'] = 1
    else:
        data_val.loc[index,'Class'] = -1