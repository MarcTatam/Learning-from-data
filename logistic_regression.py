import math
import numpy as np

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
        total += actual*math.log(predicted)+(1-y)*math.log((1-predicted))
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

def gradient(observations:list, predictions: list, actuals:list):
    observation_count = len(observations)
    total_loss = 0
    for item in range(len(predictions)):
        x_value= 0
        for j in range(len(observations[0])):
            x_value += observatios[item][j] **2
        x_value = math.sqrt(x_value)
        total_loss += (prediction[item] - actual[item])*x_value
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
                predictions.append(predict(item),weights)
            weights[i] = weights[i] - learning_rate*gradient(observations,predictions,actuals)
            current_iter -=1
    return weights
