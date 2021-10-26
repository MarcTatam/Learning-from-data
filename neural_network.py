import numpy as np

def sigmoid(z):
    """Calculates the sigmoid of a given input

    Args
    z - weighted inputs"""
    s = 1/(1+np.exp(-z))
    return s

def cost(predicted_values, actual_values):
    """Calculates the cost of the model

    Args
    predicted_values - list of the classification predicted by the neural network
    actual_values - list of the actual classifications"""
    incorrect = 0
    for i in range(0,len(predicted_values)):
        if predicted_values[i] != actual_values[i]:
            incorrect += 1
    return incorrect
