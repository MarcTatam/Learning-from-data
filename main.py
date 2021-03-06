from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import logistic_regression as lr
import weighted_knn as knn
import model_evaluation as me

def open_files()-> pd.DataFrame:
    """ Opens the files 

    Returns a dataframe with all the data from the CSV files
    """
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

def select_rows(frame: pd.DataFrame)->(pd.DataFrame,pd.DataFrame):
    """Separates the training and test data

    Args
    Frame - a dataframe representing all the data

    Returns two frames, one representing the training data and the other the test data
    """
    train, test = train_test_split(frame, test_size=0.2)
    return train, test

def save_files(train_frame, test_frame):
    """ Saves the test and training data to their own files

    Args
    train_frame - dataframe representing the training data
    test_frame - dataframe representing the test data
    """
    train_frame.to_csv("train.csv")
    test_frame.to_csv("test.csv")

def open_train()->pd.DataFrame:
    """ Opens the training data

    returns a dataframe representing the training data
    """
    return pd.read_csv("train.csv", index_col=None, header=0)

def open_test()->pd.DataFrame:
    """ Opens the training data

    returns a dataframe representing the test data
    """
    return pd.read_csv("test.csv", index_col=None, header=0)

def normalise_test(mins: [float], maxs: [float])->([[float]],[int]):
    """ Normalises the data in the test set
    
    Args
    mins - list containg the minimum value for each column
    maxs - list containg the maximum value for each column

    Returns a list containing a list of lists of floats representing each row in the data in normalised form and a list of classifications
    """
    test_data = []
    frame = open_test()
    formatted_rows, actuals = lr.format_files(frame)
    for j in range(len(formatted_rows[0])):
        row = []
        for i in range(0,len(mins)):
             row.append((formatted_rows[i][j]-mins[i])/(maxs[i]-mins[i]))
        test_data.append(row)
    return test_data, actuals

def row_to_point(test_data:[[float]])->[knn.Weighted_Point]:
    """Converts the data from row format to weighted point format
    
    Args
    test_data - data in row format
    
    Returns a list of points in space for knn"""
    points = []
    for row in test_data:
        point = knn.Weighted_Point(row[0],row[1],row[2],row[3],row[4],row[5])
        points.append(point)
    return points


def train_logistic(learning_rate, epochs)->[float]:
    """Trains the logistic model

    Args
    learning_rate - learning rate to use
    Epochs - number of iterations to train the model.

    returns a list of weights, the normalised rows, the min and max values for normalisation and the rows actual classification
    """
    formatted_rows, actuals = lr.format_files(open_train())
    normalised_rows, mins, maxs = lr.normalise(formatted_rows)
    weights = lr.optimise(normalised_rows, actuals, [0,0,0,0,0,0,0], learning_rate, max_iter = epochs)
    return weights, normalised_rows, mins, maxs, actuals

def train_knn():
    """Trains the KNN model

    returns a list of normalised points, their actual classification and the min and max values for normalisation
    """
    formatted_rows, actuals = knn.format_files(open_train())
    normalised_points, mins, maxs = knn.normalise(formatted_rows, actuals)
    return normalised_points, actuals, mins, maxs

def test_logistic(normalised_rows, actuals, weights)->[int]:
    """!!!!!DEPRECATED!!!!!
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
    """!!!!!DEPRECATED!!!!!
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
    """Graphs ROC curve for logistic regression using the training data
    
    Args
    weights - the weights for logistic regression classification
    points - the rows used in the training data
    actuals - the classification of the rows"""
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
    ax.plot([0, 1], [0, 1],'r--')
    plt.title("ROC Curve for Training Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def graph_roc_lr_test(weights, points, actuals):
    """Graphs ROC curve for logistic regression using the test data
    
    Args
    weights - the weights for logistic regression classification
    points - the test rows
    actuals - the classification of the rows"""
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
    ax.plot([0, 1], [0, 1],'r--')
    plt.title("ROC Curve for Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def graph_roc_knn_train(points, actuals):
    """Graphs ROC curve for knn using the training data
    
    Args
    points - the points used in the training data
    actuals - the classification of the points"""
    y_axis = []
    x_axis = []
    for i in range(1,100):
        threshold = 1/i
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(points)):
            classification = knn.knn_classify(points[j],points,15)
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
    ax.plot([0, 1], [0, 1],'r--')
    plt.title("ROC Curve for Training Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def graph_roc_knn_test(points, classified_points, actuals):
    """Graphs ROC curve for knn using the test data
    
    Args
    points - the points used in the test data
    classified_points - the classsified points
    actuals - the classification of the test points"""
    y_axis = []
    x_axis = []
    classifis = []
    for i in range(0,101):
        print(i)
        if i == 101:
            threshold = -0.1
        elif i == 0:
            threshold = 1.1
        else:
            threshold = 1/i
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(points)):
            classification = knn.knn_classify(points[j],classified_points,15)
            classifis.append(classification)
            if classification > 1:
                print(classification)
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
    ax.plot([0, 1], [0, 1],'r--')
    plt.title("ROC Curve for Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def knn_f1(trained_points, test_points, actuals_train, actuals_test):
    """Graphs the performance of knn as K changes

    Args
    trained_points - points in the training set
    test_points - points in the test set
    actuals_train - classification of training points
    actuals_test - classification of test points
    """
    x_values1  = []
    y_values1  = []
    for i in range(1,31):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(trained_points)):
            classification = knn.knn_classify(trained_points[j],trained_points,i)
            if classification > 1:
                print(classification)
            if classification >= 0.5 and actuals_train[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_train[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_train[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values1.append(i)
        y_values1.append(me.f1_score(tp,fp,fn))
    x_values2 = []
    y_values2 = []
    for i in range(1,31):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(test_points)):
            classification = knn.knn_classify(test_points[j],trained_points,i)
            if classification > 1:
                print(classification)
            if classification >= 0.5 and actuals_test[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_test[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_test[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values2.append(i)
        y_values2.append(me.f1_score(tp,fp,fn))

    ax = plt.axes()
    ax.plot(x_values1, y_values1, "g")
    ax.plot(x_values2, y_values2, "r")
    plt.title("F Score as K increases")
    plt.xlabel("K Value")
    plt.ylabel("F Score")
    plt.show()

def lr_f1_lr(test_points, actuals_test):
    """Graphs f1 as the learning rate changes

    Args
    test_points - Test set
    actuals_test - classification of test set
    """
    x_values1  = []
    y_values1  = []
    x_values2 = []
    y_values2 = []
    for i in range(1,21):
        print(i)
        learning_rate = i/200
        weights, trained_points, mins, maxs, actuals_train = train_logistic(learning_rate,1000)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(trained_points)):
            classification = lr.predict(trained_points[j],weights)
            if classification >= 0.5 and actuals_train[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_train[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_train[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values1.append(learning_rate)
        y_values1.append(me.f1_score(tp,fp,fn))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(test_points)):
            classification = lr.predict(test_points[j],weights)
            if classification > 1:
                print(classification)
            if classification >= 0.5 and actuals_test[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_test[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_test[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values2.append(learning_rate)
        y_values2.append(me.f1_score(tp,fp,fn))
    ax = plt.axes()
    ax.plot(x_values1, y_values1, "g")
    ax.plot(x_values2, y_values2, "r")
    plt.title("F1 Score as Learning Rate increases")
    plt.xlabel("Learning Rate")
    plt.ylabel("F1 Score")
    plt.show()

def lr_f1_epochs(test_points, actuals_test):
    """Graphs f1 as the number of iterations changes

    Args
    test_points - Test set
    actuals_test - classification of test set
    """
    x_values1  = []
    y_values1  = []
    x_values2 = []
    y_values2 = []
    for i in range(1,30):
        print(i)
        epochs= i*100
        weights, trained_points, mins, maxs, actuals_train = train_logistic(0.0005,epochs)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(trained_points)):
            classification = lr.predict(trained_points[j],weights)
            if classification >= 0.5 and actuals_train[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_train[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_train[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values1.append(epochs)
        y_values1.append(me.f1_score(tp,fp,fn))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(test_points)):
            classification = lr.predict(test_points[j],weights)
            if classification > 1:
                print(classification)
            if classification >= 0.5 and actuals_test[j] >= 0.5:
                tp += 1
            elif classification >= 0.5 and actuals_test[j] < 0.5:
                fp += 1
            elif classification < 0.5 and actuals_test[j] < 0.5:
                tn += 1
            else:
                fn += 1
        x_values2.append(epochs)
        y_values2.append(me.f1_score(tp,fp,fn))
    ax = plt.axes()
    ax.plot(x_values1, y_values1, "g")
    ax.plot(x_values2, y_values2, "r")
    plt.title("F1 Score as Number of Iterations Increases")
    plt.xlabel("Iterations")
    plt.ylabel("F1 Score")
    plt.show()

def comparison(test_rows, test_actuals):
    """Calculates perfomence metrics

    Args
    test_rows - Test set
    test_actuals - classification of test set
    """
    test_points = row_to_point(test_rows)
    trained_points, actuals_train, mins, maxs = train_knn()
    weights, trained_rows, mins, max, actuals_train = train_logistic(0.0005,1000)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(trained_rows)):
        classification = lr.predict(trained_rows[j],weights)
        if classification >= 0.5 and actuals_train[j] >= 0.5:
            tp += 1
        elif classification >= 0.5 and actuals_train[j] < 0.5:
            fp += 1
        elif classification < 0.5 and actuals_train[j] < 0.5:
            tn += 1
        else:
            fn += 1
    print("------Logistic Training------")
    print("Accuracy : " + str(me.accuracy(tp+tn,tp+tn+fp+fn)))
    print("Precision : " + str(me.precision(tp,fp)))
    print("Recall : " + str(me.recall(tp,fn)))
    print("F1 : " +str(me.f1_score(tp,fp,fn)))
    print("TPR : " + str(me.true_positive_rate(tp,fn)))
    print("FPR : " + str(me.false_positive_rate(fp,tn)))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(test_rows)):
        classification = lr.predict(test_rows[j],weights)
        if classification >= 0.5 and actuals_test[j] >= 0.5:
            tp += 1
        elif classification >= 0.5 and actuals_test[j] < 0.5:
            fp += 1
        elif classification < 0.5 and actuals_test[j] < 0.5:
            tn += 1
        else:
            fn += 1
    print("------Logistic Test------")
    print("Accuracy : " + str(me.accuracy(tp+tn,tp+tn+fp+fn)))
    print("Precision : " + str(me.precision(tp,fp)))
    print("Recall : " + str(me.recall(tp,fn)))
    print("F1 : " +str(me.f1_score(tp,fp,fn)))
    print("TPR : " + str(me.true_positive_rate(tp,fn)))
    print("FPR : " + str(me.false_positive_rate(fp,tn)))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(trained_points)):
        classification = knn.knn_classify(trained_points[j],trained_points,5)
        if classification > 1:
            print(classification)
        if classification >= 0.5 and actuals_train[j] >= 0.5:
            tp += 1
        elif classification >= 0.5 and actuals_train[j] < 0.5:
            fp += 1
        elif classification < 0.5 and actuals_train[j] < 0.5:
            tn += 1
        else:
            fn += 1
    print("------KNN Train------")
    print("Accuracy : " + str(me.accuracy(tp+tn,tp+tn+fp+fn)))
    print("Precision : " + str(me.precision(tp,fp)))
    print("Recall : " + str(me.recall(tp,fn)))
    print("F1 : " +str(me.f1_score(tp,fp,fn)))
    print("TPR : " + str(me.true_positive_rate(tp,fn)))
    print("FPR : " + str(me.false_positive_rate(fp,tn)))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(test_points)):
        classification = knn.knn_classify(test_points[j],trained_points,5)
        if classification > 1:
            print(classification)
        if classification >= 0.5 and actuals_test[j] >= 0.5:
            tp += 1
        elif classification >= 0.5 and actuals_test[j] < 0.5:
            fp += 1
        elif classification < 0.5 and actuals_test[j] < 0.5:
            tn += 1
        else:
            fn += 1
    print("------KNN Test------")
    print("Accuracy : " + str(me.accuracy(tp+tn,tp+tn+fp+fn)))
    print("Precision : " + str(me.precision(tp,fp)))
    print("Recall : " + str(me.recall(tp,fn)))
    print("F1 : " +str(me.f1_score(tp,fp,fn)))
    print("TPR : " + str(me.true_positive_rate(tp,fn)))
    print("FPR : " + str(me.false_positive_rate(fp,tn)))
    
def knn_graphs(trained_points, test_points):
    """Graphs knn classification of points
    
    Args
    trained_points - Training set
    test_points - Test set
    """
    training_wins_for  = []
    training_wins_against  = []
    training_loss_for  = []
    training_loss_against  = []
    for j in range(len(trained_points)):
        classification = knn.knn_classify(trained_points[j],trained_points,5)
        if classification >= 0.5:
            training_wins_for.append(trained_points[j].dimension5)
            training_wins_against.append(trained_points[j].dimension6)
        else:
            training_loss_for.append(trained_points[j].dimension5)
            training_loss_against.append(trained_points[j].dimension6)
    test_wins_for  = []
    test_wins_against  = []
    test_loss_for  = []
    test_loss_against  = []
    for j in range(len(test_points)):
        classification = knn.knn_classify(test_points[j],trained_points,5)
        if classification >= 0.5:
            test_wins_for.append(test_points[j].dimension5)
            test_wins_against.append(test_points[j].dimension6)
        else:
            test_loss_for.append(test_points[j].dimension5)
            test_loss_against.append(test_points[j].dimension6)

    ax = plt.axes()
    print(training_wins_for)
    print(training_wins_against)
    train_win = ax.scatter(training_wins_for, training_wins_against, c="g")
    train_loss = ax.scatter(training_loss_for, training_loss_against, c="r")
    test_win = ax.scatter(test_wins_for, test_wins_against, c="b")
    test_loss = ax.scatter(test_loss_for, test_loss_against, c="orange")
    ax.legend([train_win, train_loss, test_win, test_loss], ["Train Win", "Train Loss","Test Win", "Test Loss"])
    plt.ylabel("Normalised Passes for")
    plt.xlabel("Normalised Passes against")
    plt.show()

def knn_graphs_correctness(trained_points, actuals_train, test_points, actuals_test):
    """Graphs knn classification correctness of points
    
    Args
    trained_points - Training set
    actuals_train - Actual classification of training data
    test_points - Test set
    actuals_test - Actual classification of test data
    """
    training_correct_for  = []
    training_correct_against  = []
    training_incorrect_for  = []
    training_incorrect_against  = []
    for j in range(len(trained_points)):
        classification = knn.knn_classify(trained_points[j],trained_points,5)
        if classification >= 0.5 and actuals_train[j] >= 0.5:
            training_correct_for.append(trained_points[j].dimension5)
            training_correct_against.append(trained_points[j].dimension6)
        elif classification >= 0.5 and actuals_train[j] < 0.5:
            training_incorrect_for.append(trained_points[j].dimension5)
            training_incorrect_against.append(trained_points[j].dimension6)
        elif classification < 0.5 and actuals_train[j] < 0.5:
            training_correct_for.append(trained_points[j].dimension5)
            training_correct_against.append(trained_points[j].dimension6)
        else:
            training_incorrect_for.append(trained_points[j].dimension5)
            training_incorrect_against.append(trained_points[j].dimension6)
    test_correct_for  = []
    test_correct_against  = []
    test_incorrect_for  = []
    test_incorrect_against  = []
    for j in range(len(test_points)):
        classification = knn.knn_classify(test_points[j],trained_points,5)
        if classification >= 0.5 and actuals_test[j] >= 0.5:
            test_correct_for.append(test_points[j].dimension5)
            test_correct_against.append(test_points[j].dimension6)
        elif classification >= 0.5 and actuals_test[j] < 0.5:
            test_incorrect_for.append(test_points[j].dimension5)
            test_incorrect_against.append(test_points[j].dimension6)
        elif classification < 0.5 and actuals_test[j] < 0.5:
            test_correct_for.append(test_points[j].dimension5)
            test_correct_against.append(test_points[j].dimension6)
        else:
            test_incorrect_for.append(test_points[j].dimension5)
            test_incorrect_against.append(test_points[j].dimension6)
    ax = plt.axes()
    train_correct = ax.scatter(training_correct_for, training_correct_against, c="g")
    train_incorrect = ax.scatter(training_incorrect_for, training_incorrect_against, c="r")
    test_correct = ax.scatter(test_correct_for, test_correct_against, c="b")
    test_incorrect = ax.scatter(test_incorrect_for, test_incorrect_against, c="orange")
    #ax.plot([0, 1], [0, 1],'r--')
    ax.legend([train_correct, train_incorrect, test_correct, test_incorrect], ["Train Correct", "Train Incorrect","Test Correct", "Test Incorrect"])
    plt.ylabel("Normalised Passes for")
    plt.xlabel("Normalised Passes against")
    plt.show()

def lr_graphs(trained_rows, test_rows, weights):
    """Graphs logistic regression classification of points
    
    Args
    trained_points - Training set
    test_points - Test set
    weights - Weights for regression
    """
    training_wins_for  = []
    training_wins_against  = []
    training_loss_for  = []
    training_loss_against  = []
    for j in range(len(trained_rows)):
        classification = lr.predict(trained_rows[j], weights)
        if classification >= 0.5:
            training_wins_for.append(trained_rows[j][4])
            training_wins_against.append(trained_rows[j][5])
        else:
            training_loss_for.append(trained_rows[j][4])
            training_loss_against.append(trained_rows[j][5])
    test_wins_for  = []
    test_wins_against  = []
    test_loss_for  = []
    test_loss_against  = []
    for j in range(len(test_rows)):
        classification = lr.predict(test_rows[j], weights)
        if classification >= 0.5:
            test_wins_for.append(test_rows[j][4])
            test_wins_against.append(test_rows[j][5])
        else:
            test_loss_for.append(test_rows[j][4])
            test_loss_against.append(test_rows[j][5])
    ax = plt.axes()
    print(training_wins_for)
    print(training_wins_against)
    train_win = ax.scatter(training_wins_for, training_wins_against, c="g")
    train_loss = ax.scatter(training_loss_for, training_loss_against, c="r")
    test_win = ax.scatter(test_wins_for, test_wins_against, c="b")
    test_loss = ax.scatter(test_loss_for, test_loss_against, c="orange")
    ax.legend([train_win, train_loss, test_win, test_loss], ["Train Win", "Train Loss","Test Win", "Test Loss"])
    plt.ylabel("Normalised Passes for")
    plt.xlabel("Normalised Passes against")
    plt.show()

def lr_graphs_correctness(trained_rows, actuals_train, test_rows, actuals_test,weights):
    """Graphs logistic regression classification correctness of points
    
    Args
    trained_points - Training set
    actuals_train - Actual classification of training data
    test_points - Test set
    actuals_test - Actual classification of test data
    weights - Weights for regression
    """
    training_correct_for  = []
    training_correct_against  = []
    training_incorrect_for  = []
    training_incorrect_against  = []
    for j in range(len(trained_points)):
        classification = lr.predict(trained_rows[j], weights)
        if classification >= 0.5 and actuals_train[j] >= 0.5:
            training_correct_for.append(trained_rows[j][4])
            training_correct_against.append(trained_rows[j][5])
        elif classification >= 0.5 and actuals_train[j] < 0.5:
            training_incorrect_for.append(trained_rows[j][4])
            training_incorrect_against.append(trained_rows[j][5])
        elif classification < 0.5 and actuals_train[j] < 0.5:
            training_correct_for.append(trained_rows[j][4])
            training_correct_against.append(trained_rows[j][5])
        else:
            training_incorrect_for.append(trained_rows[j][4])
            training_incorrect_against.append(trained_rows[j][5])
    test_correct_for  = []
    test_correct_against  = []
    test_incorrect_for  = []
    test_incorrect_against  = []
    for j in range(len(test_rows)):
        classification = lr.predict(test_rows[j], weights)
        if classification >= 0.5 and actuals_test[j] >= 0.5:
            test_correct_for.append(test_rows[j][4])
            test_correct_against.append(test_rows[j][5])
        elif classification >= 0.5 and actuals_test[j] < 0.5:
            test_incorrect_for.append(test_rows[j][4])
            test_incorrect_against.append(test_rows[j][5])
        elif classification < 0.5 and actuals_test[j] < 0.5:
            test_correct_for.append(test_rows[j][4])
            test_correct_against.append(test_rows[j][5])
        else:
            test_incorrect_for.append(test_rows[j][4])
            test_incorrect_against.append(test_rows[j][5])
    ax = plt.axes()
    train_correct = ax.scatter(training_correct_for, training_correct_against, c="g")
    train_incorrect = ax.scatter(training_incorrect_for, training_incorrect_against, c="r")
    test_correct = ax.scatter(test_correct_for, test_correct_against, c="b")
    test_incorrect = ax.scatter(test_incorrect_for, test_incorrect_against, c="orange")
    #ax.plot([0, 1], [0, 1],'r--')
    ax.legend([train_correct, train_incorrect, test_correct, test_incorrect], ["Train Correct", "Train Incorrect","Test Correct", "Test Incorrect"])
    plt.ylabel("Normalised Passes for")
    plt.xlabel("Normalised Passes against")
    plt.show()


if __name__ == "__main__":
    weights, trained_points, mins, maxs, actuals_train  = train_logistic(0.0005, 1000)
    normalised_rows, actuals_test =  normalise_test(mins, maxs)
    print(weights)
    #normalised_points = row_to_point(normalised_rows)
    #lr_f1_lr(normalised_rows, actuals_test)