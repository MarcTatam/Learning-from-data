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
    return frame

if __name__ == "__main__":
    print(open_files())