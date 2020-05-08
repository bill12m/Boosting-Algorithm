import pandas as pd
import numpy as np
import subprocess as sp
#from pprint import pprint

sp.call('clear', shell = True)

#Build training dataframe and calculate original probabilities
data_train = pd.DataFrame(np.array([[0.5,-1],[3.0,-1],[4.5,1],[4.6,1],[4.9,1],
                                    [5.2,-1],[5.3,-1],[5.5,1],[7.0,-1],[9.5,-1]])
                          , columns = ['x','Class'])
data_train['Probabilities'] = np.zeros(shape = (len(data_train)))
for index in data_train.index:
    data_train.loc[index,'Probabilities'] += 1/(len(data_train))
#data_train['Class'] = data_train['y']
#del(data_train['y'], index)
epochs = 0
model = []
model_weights = []

#Functions for calculating variables in the program
def probability_next(error,probability,classified):
    #probability_next = 0
    if classified == False:
        probability_next = probability/(2*error)
    else:
        probability_next = probability/(2*(1-error))
    return probability_next
def calculate_weight(error):
    alpha = 0.5*np.log((1-error)/error)
    return alpha
def calculate_error(probabilities):
    error = np.sum(probabilities)
    return error   
def calculate_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    targets = df[Class].unique()
    for target in targets:
        probability = df[Class].value_counts()[target]/len(df[Class])
        entropy += -probability*np.log2(probability)
    return entropy

#Running program for 100 epochs maximum. There will be other stopping
#conditions within the while loop
while epochs < 100:
    #Construct a new training set with the updated probabilities.
    new_data_train = data_train.iloc[:,0:2].sample(n = len(data_train), replace = True, 
                                                   weights = data_train['Probabilities']).sort_values(
                                                       by = ['x']).reset_index()
    del new_data_train['index']

    #Construct a decision stump for the 1-dimensional data set.
    #Select cut-off point by finding the split which maximizes the information 
    #gain calculated using entropy
    
    #Lists to contain the respective information gains and splitting
    #conditions for each potential node-split
    information_gain = []
    splits = []
    #Check to see if all instances have the same classification
    uniques = new_data_train['Class'].nunique()
    if  uniques == 1:
        split = 0.1
        model.append(split)
    #If they don't have the same classification, check to see which split
    #offers the best information gain and save that as our model for this
    #epoch
    else:
        for instance in new_data_train.index:
            split = new_data_train.loc[instance,'x'] + 0.01
            splits.append(split)
            #New dataframes for each half of the training set created by
            #the splitting condition
            child_1 = pd.DataFrame(new_data_train[new_data_train['x'] <= split])
            child_2 = pd.DataFrame(new_data_train[new_data_train['x'] > split])
            weighted_average = (len(child_1)/len(new_data_train)) * calculate_entropy(child_1)
            + (len(child_2)/len(new_data_train) * calculate_entropy(child_2))
            information_gain.append(1 - weighted_average)
            
        split = splits[np.argmax(np.asarray(information_gain))]
        model.append(split)
    
    #Run the current model on the original training data to determine its
    #accuracy
    data_train['Correctly Classified'] = np.zeros(shape = len(data_train))
    for index in data_train.index:
        predicted_class = 0
        if data_train.loc[index,'x'] < split:
            predicted_class -= 1
        else:
            predicted_class += 1
        if data_train.loc[index,'Class'] == predicted_class:
            data_train.loc[index,'Correctly Classified'] = 1
        else:
            data_train.loc[index,'Correctly Classified'] = 0
    del index    
    #Calculate the error of the current model
    error = calculate_error(data_train['Probabilities'][data_train['Correctly Classified'] == 0])
    #If the model is less than 50% accurate, scrap it
    if error >= 0.5:
        del model[-1]
    else:
        #Calculate the weight for the current model
        model_weights.append(calculate_weight(error))
    
        #Calculate the new probabilities based on the misclassifications    
        for index in data_train.index:
            if data_train.loc[index,'Correctly Classified'] == 1:
                data_train.loc[index, 'Probabilities'] = probability_next(
                    error = error,
                    probability  = data_train.loc[index,'Probabilities'],
                    classified = True)
            else:
                data_train.loc[index,'Probabilities'] = probability_next(
                    error = error,
                    probability = data_train.loc[index,'Probabilities'],
                    classified = False)

    #del(child_1, child_2, information_gain, instance, uniques, splits, weighted_average)
    epochs +=1
    
                                       
