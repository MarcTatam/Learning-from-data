from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os

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
    print(train, test)

if __name__ == "__main__":
    frame = open_files()
    select_rows(frame)