import math
import json
import os
import glob
import numpy as np
import pandas as pd

def sigmoid(z:float)->float:
    """Calculates the sigmoid of a given input

    Args
    z - weighted inputs"""
    print(z)
    s = 1/(1+np.exp(-z))
    return s

def cost(predicted_values:list, actual_values:list)->int:
    """Calculates the cost of the model

    Args
    predicted_values - list of the classification predicted by the neural network
    actual_values - list of the actual classifications"""
    incorrect = 0
    for i in range(0,len(predicted_values)):
        if predicted_values[i] != actual_values[i]:
            incorrect += 1
    return incorrect

def calculate_input(values:list, weights:list)->float:
    """Calculates the input for the sigmoid function
    
    Args
    values - list of the unweighted values of the inputs
    wieghts - list of the weights
    
    Returns
    float representing the z value of the sigmoid function"""
    total = 0
    for i in range(len(weights)):
        if i == len(values):
            total += weights[i]
        else:
            total += values[i]*weights[i]
    return total

def log_likelihood(predicted_probablities:list, actual_probabilities:list)->float:
    """Calculates the maximum likelihood estimation

    Args
    predicted_probabilities - list of the probabilities of each observation
    actual_probabilities - actual classification (1 for win 0 for loss)
    
    Returns
    float representing the maximum likelihood estimation"""
    total = 0
    for i in range(len(actual_probabilities)):
        predicted = predicted_probablities[i]
        actual = actual_probabilities[i]
        total += actual*math.log(predicted)+(1-actual)*math.log((1-predicted))
    return -total/len(actual_probabilities)

def predict(row:list, weights)->int:
    """Predicts the outcome for a given row

    Args
    row - the row to predict the outcome for
    weights - the weights to use for the prediction
    
    Returns
    An integer representing a predicted outcome (1 for win, 0 for loss)"""
    z = calculate_input(row, weights)
    probability = sigmoid(z)
    if probability > 0.5:
        return 1
    else:
        return 0

def gradients(observations:list, predictions: list, actuals:list):
    observation_count = len(observations)
    dw = 0
    db = 0
    for i in range(len(observations)):
        inaccuracy = predictions[i]-actual[i]
        dw += (inaccuracy)/observation_count #to do dot product
        db += (inaccuracy)/observation_count
        
    return -total_loss/observation_count


def optimise(observations:list, actuals:list, weights:list, index:int, learning_rate:float, max_iter:int=100)->list:
    """Optimises a given weight

    Args
    observations - List of the observations
    actuals - actual class of each observation
    weights - the list of weights
    index - the weight to optimise
    learning_rate - the learning rate
    max_iter - the maximum iterations
    """
    for i in range(len(weights)):
        current_iter = max_iter
        while current_iter > 0:
            predictions = []
            for item in observations:
                predictions.append(predict(item,weights))
            weights[i] = weights[i] - learning_rate*gradient(observations,predictions,actuals)
            current_iter -=1
    return weights

def open_files():
    current_path = os.path.dirname(os.path.realpath(__file__))
    all_files_path = glob.glob(current_path + "/*.csv")

    file_list = []

    for filename in all_files_path:
        dataframe = pd.read_csv(filename, index_col=None, header=0)
        file_list.append(dataframe)

    frame = pd.concat(file_list, axis=0, ignore_index=True)
    frame = frame[frame.h_a == "h"]
    frame = frame[frame.result != "d"]
  
    return frame

def format_files(frame)->([[float,float,int,int,int,int]],[str]):
    rows = []
    actual = []
    for row in frame.itertuples():
        parsed_format = json.loads(row[6].replace("'","\""))
        parsed_formatA = json.loads(row[7].replace("'","\""))
        this_row = [row[2],row[3],row[8],row[9],parsed_format['att'],parsed_formatA['att']]
        rows.append(this_row)
        if row[13] == "w":
            actual.append(1)
        else:
            actual.append(0)
    return rows, actual

def test(points,classifis):
    weights = [0,0,0,0,0,0]
    for i in range(6):
        weights = optimise(points,classifis,weights,i,0.1)
    print(weights)


if __name__ == "__main__":
    frame = open_files()
    rows,actual = format_files(frame)
    test(rows,actual)
