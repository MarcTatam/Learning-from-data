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
        if i == len(weights)-1:
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
    return sigmoid(z)

def optimise(observations:list, actuals:list, weights:list, learning_rate:float, max_iter:int=1000)->list:
    """Optimises a set of weights

    Args
    observations - List of the observations
    actuals - actual class of each observation
    weights - the list of weights
    index - the weight to optimise
    learning_rate - the learning rate
    max_iter - the maximum iterations
    """
    for j in range(max_iter):
        mse = 0
        for row_ind in range(len(observations)):
            row = observations[row_ind]
            prediction = predict(row, weights)
            error = actuals[row_ind] - prediction
            mse += error**2
            for i in range(len(weights)-1):
                weights[i] = weights[i] + learning_rate*error*prediction*(1-prediction)*row[i]
            weights[len(weights)-1] = weights[len(weights)-1] + learning_rate * error * prediction * (1.0 - prediction)
    return weights

def open_files():
    """Opens the files
    
    returns a frame holding the data"""
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

def format_files(frame)->([[float],[float],[int],[int],[int],[int]],[int]):
    """Formats the files into a workable format
    
    Args
    frame - frame holding data
    
    returns a list with each column and the actual classification
    """
    columns =[[],[],[],[],[],[]]
    actual = []
    for row in frame.itertuples():
        parsed_format = json.loads(row[6].replace("'","\""))
        parsed_formatA = json.loads(row[7].replace("'","\""))
        columns[0].append(row[2])
        columns[1].append(row[3])
        columns[2].append(row[8])
        columns[3].append(row[9])
        columns[4].append(parsed_format['att'])
        columns[5].append(parsed_formatA['att'])
        if row[13] == "w":
            actual.append(1)
        else:
            actual.append(0)
    return columns, actual

def normalise(points:[[float],[float],[int],[int],[int],[int]])->[[float,float,float,float,float,float]]:
    """Normalises the data
    
    Args
    points - points to normalise
    actuals - True classification of points"""
    maxs = [max(points[0]),max(points[1]),max(points[2]),max(points[3]),max(points[4]),max(points[5])]
    mins = [min(points[0]),min(points[1]),min(points[2]),min(points[3]),min(points[4]),min(points[5])]
    rows = []
    for i in range(len(points[0])):
        this_row1 = (points[0][i]-mins[0])/(maxs[0]-mins[0])
        this_row2 = (points[1][i]-mins[1])/(maxs[1]-mins[1])
        this_row3 = (points[2][i]-mins[2])/(maxs[2]-mins[2])
        this_row4 = (points[3][i]-mins[3])/(maxs[3]-mins[3])
        this_row5 = (points[4][i]-mins[4])/(maxs[4]-mins[4])
        this_row6 = (points[5][i]-mins[5])/(maxs[5]-mins[5])
        this_row = [this_row1,this_row2,this_row3,this_row4,this_row5,this_row6]
        rows.append(this_row)
    return rows,mins,maxs

def test(points,classifis):
    weights = [1,-1,1,-1,1,-1,0]
    weights = optimise(points,classifis,weights,0.15)
    correct = 0
    incorrect = 0
    for row_ind in range(len(points)):
        prediction = predict(points[row_ind],weights)
        if prediction >= 0.5:
            prediction = 1
        else:
            prediction  = 0
        if prediction == classifis[row_ind]:
            correct += 1
        else:
            incorrect += 1
    return(correct/(correct+incorrect))
        

    


if __name__ == "__main__":
    frame = open_files()
    columns,actual = format_files(frame)
    rows,mins,maxs = normalise(columns)
    print(test(rows,actual))
