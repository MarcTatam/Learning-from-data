import matplotlib.pyplot as plt

def precision(true_positives : int, false_positives : int)->float:
    """Calculates the precision of a classifier

    Args
    true_positive - an integer value for the number of correctly clssified positives
    false_positives - an integer value for the number of incorrectly classified positives

    Returns
    A float representing the precision of the classifier
    """
    return true_positives/(true_positives+false_positives)

def recall(true_positives : int, false_negatives :int)->float:
    """Calculates the recall of a classifier

    Args
    true_positive - an integer value for the number of correctly clssified positives
    false_negatives - an integer value for the number of incorrectly classified negatives

    Returns
    A float representing the recall of the classifier
    """
    return true_positives/(true_positives+false_negatives)

def f1_score(true_positives : int, false_positives : int, false_negatives :int)->float:
    """Calculates the F1 score of a classifier

    Args
    true_positive - an integer value for the number of correctly clssified positives
    false_positives - an integer value for the number of incorrectly classified positives
    false_negatives - an integer value for the number of incorrectly classified negatives

    Returns
    A float representing the F1 score of the classifier
    """
    precision = precision(true_positives, false_positives)
    recall = recall(true_positives, false_negatives)
    return 2*((precision*recall)/(precision+recall))

def true_positive_rate(true_positives : int, false_negatives :int):
    """Calculates the true positive rate of a classifier

    Args
    true_positive - an integer value for the number of correctly clssified positives
    false_negatives - an integer value for the number of incorrectly classified negatives

    Returns
    A float representing the true positive rate of the classifier
    """
    if true_positives == 0 and false_negatives == 0:
        return 0
    return true_positives/(true_positives+false_negatives)

def false_positive_rate(false_positives : int, true_negatives :int):
    """Calculates the true positive rate of a classifier

    Args
    false_positive - an integer value for the number of incorrectly clssified positives
    true_negatives - an integer value for the number of correctly classified negatives

    Returns
    A float representing the true positive rate of the classifier
    """
    if false_positives == 0 and true_negatives == 0:
        return 0
    return false_positives/(false_positives + true_negatives)

def f_score_comparison(knn_list : list, logisitc_list : list):
    """Plots a graph comparing the f scores of the two different classification techniques

    Args
    knn_list list of the results for knn in the order true positives, false positives, true negatives, false negatives
    knn_list list of the results for the other method in the order true positives, false positives, true negatives, false negatives
    """