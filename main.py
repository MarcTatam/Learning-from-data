from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import logistic_regression as lr
import weighted_knn as knn
import model_evaluation as me

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

def select_rows(frame):
    train, test = train_test_split(frame, test_size=0.2)
    return train, test

def save_files(train_frame, test_frame):
    train_frame.to_csv("train.csv")
    test_frame.to_csv("test.csv")

def open_train()->pd.DataFrame:
     return pd.read_csv("train.csv", index_col=None, header=0)

def open_test()->pd.DataFrame:
     return pd.read_csv("test.csv", index_col=None, header=0)

def normalise_test(mins, maxs):
    test_data = []
    frame = open_test()
    formatted_rows, actuals = lr.format_files(frame)
    for j in range(len(formatted_rows[0])):
        row = []
        for i in range(0,len(mins)):
             row.append((formatted_rows[i][j]-mins[i])/(maxs[i]-mins[i]))
        test_data.append(row)
    return test_data, actuals


def train_logistic()->[float]:
    formatted_rows, actuals = lr.format_files(open_train())
    normalised_rows, mins, maxs = lr.normalise(formatted_rows)
    weights = lr.optimise(normalised_rows, actuals, [0,0,0,0,0,0,0], 0.05)
    return weights, normalised_rows, mins, maxs, actuals

def train_knn():
    formatted_rows, actuals = knn.format_files(open_train())
    normalised_points, mins, maxs = knn.normalise(formatted_rows)
    return normalised_points

def test_logistic(normalised_rows, actuals, weights)->[int]:
    """
    1 - True Positive
    2 - False Positive
    3 - True Negative
    4 - False Negative
    """
    types = []
    for i in range(len(normalised_rows)):
        classification = lr.predict(normalised_rows[i],weights)
        if classification >= 0.5 and actuals[i] >= 0.5:
            types.append(1)
        elif classification >= 0.5 and actuals[i] < 0.5:
            types.append(2)
        elif classification < 0.5 and actuals[i] < 0.5:
            types.append(3)
        else:
            types.append(4)
    return types

def test_knn(points, actuals, weights)->[int]:
    """
    1 - True Positive
    2 - False Positive
    3 - True Negative
    4 - False Negative
    """
    types =[]
    for i in range(len(points)):
        classification = knn.knn_classify(points[i],actuals, 15)
        if classification == "w" and actuals[i] == "w":
            types.append(1)
        elif classification == "w" and actuals[i] == "l":
            types.append(2)
        elif classification == "l" and actuals[i] == "l":
            types.append(3)
        else:
            types.append(4)

def graph_roc_lr_train(weights, points, actuals):
    y_axis = []
    x_axis = []
    for i in range(1,1000):
        threshold = 1/i
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(points)):
            classification = lr.predict(points[j],weights)
            if classification >= threshold and actuals[j] >= threshold:
                tp += 1
            elif classification >= threshold and actuals[j] < threshold:
                fp += 1
            elif classification < threshold and actuals[j] < threshold:
                tn += 1
            else:
                fn += 1
        y_axis.append(me.true_positive_rate(tp, fn))
        x_axis.append(me.false_positive_rate(fp, tn))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x_axis, y_axis)
    plt.title("ROC Curve for Training Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def graph_roc_lr_test(weights, points, actuals):
    y_axis = []
    x_axis = []
    for i in range(1,1000):
        threshold = 1/i
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(points)):
            classification = lr.predict(points[j],weights)
            if classification >= threshold and actuals[j] >= threshold:
                tp += 1
            elif classification >= threshold and actuals[j] < threshold:
                fp += 1
            elif classification < threshold and actuals[j] < threshold:
                tn += 1
            else:
                fn += 1
        y_axis.append(me.true_positive_rate(tp, fn))
        x_axis.append(me.false_positive_rate(fp, tn))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x_axis, y_axis)
    plt.title("ROC Curve for Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


if __name__ == "__main__":
    weights, normalised_rows, mins, maxs, actuals = train_logistic()
    #normalised_rows, actuals =  normalise_test(mins, maxs)
    graph_roc_lr_train(weights, normalised_rows, actuals)