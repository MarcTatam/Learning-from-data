from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os
import logistic_regression as lr
import weighted_knn as knn

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
    validation = frame.sample(n = 60,replace = False)
    train, test = train_test_split(frame, test_size=0.2)
    return train, test

def save_files(train_frame, test_frame):
    train_frame.to_csv("train.csv")
    test_frame.to_csv("test.csv")

def open_train()->pd.DataFrame:
     return pd.read_csv("train.csv", index_col=None, header=0)

def open_test()->pd.DataFrame:
     return pd.read_csv("train.csv", index_col=None, header=0)

def train_logistic()->[float]:
    formatted_rows, actuals = lr.format_files(open_train())
    normalised_rows, mins, maxs = lr.normalise(formatted_rows)
    weights = lr.optimise(normalised_rows, actuals, [0,0,0,0,0,0,0], 0.05)

def train_knn():
    formatted_rows, actuals = knn.format_files(open_train)
    normalised_points, mins, maxs = knn.normalise(formatted_rows)

def test_logistic(normalised_rows, actuals, weights)->[int]:
    """
    1 - True Positive
    2 - False Positive
    3 - True Negative
    4 - False Negative
    """
    types = []
    for i in range(len(normalised_rows)):
        classification = lr.predict(normalised_rows[i])
        if classification > 0.5 and actuals[i] > 0.5:
            types.append(1)

if __name__ == "__main__":
    train_logistic()