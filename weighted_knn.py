import pandas as pd
import glob
import os
import math
import json

class Weighted_Point(object):
    def __init__(self, coord1,coord2,coord3,coord4,coord5,coord6, weight, classification=None, distance=None):
        self.dimension1 = coord1
        self.dimension2 = coord2
        self.dimension3 = coord3
        self.dimension4 = coord4
        self.dimension5 = coord5
        self.dimension6 = coord6
        self.weight = weight
        self.classification = classification
        self.distance = distance

    def distance_to(self, point2)->float:
        distance1_squared = (self.dimension1-point2.dimension1)**2+(self.dimension2-point2.dimension2)**2
        distance2_squared = (self.dimension3-point2.dimension3)**2+distance1_squared
        distance3_squared = (self.dimension4-point2.dimension4)**2+distance2_squared
        distance4_squared = (self.dimension5-point2.dimension5)**2+distance3_squared
        distance5_squared = (self.dimension6-point2.dimension6)**2+distance4_squared
        return math.sqrt(distance5_squared)
    
    def set_distance(self, unclassified_point):
        self.distance = self.distance_to(unclassified_point)

    def __eq__(self, other):
        if other.distance is None or self.distance is None:
            return False
        else:
            return self.distance == other.distance

    def __gt__(self, other):
        return self.distance > other.distance
    
    def __lt__(self, other):
        return self.distance < other.distance


def knn_classify(point:Weighted_Point, classified_points:Weighted_Point, k:int)->str:
    best_points = []
    for classified_point in classified_points:
        if classified_point.classification == "w":
            classified_point.set_distance(point)
        else:
            classified_point.set_distance(point)
        if len(best_points) < k:
            best_points.append(classified_point)
            if len(best_points) == k:
                best_points.sort()
                best_points.reverse()
        elif classified_point < best_points[0]:
            i = 0
            while i < len(best_points) and classified_point < best_points[i] and i != k:
                i += 1
            if i == 0:
                best_points.pop(0)
                best_points.insert(0, classified_point)
            else:
                best_points.insert(i, classified_point)
                best_points.pop(0)
    return weighted_vote(best_points)

def weighted_vote(points:[Weighted_Point]):
    w_sum = 0
    l_sum = 0
    for point in points:
        if point.classification == "w":
            w_sum += point.weight
        else:
            l_sum += point.weight
    if l_sum < w_sum:
        return "w"
    else:
        return "l"

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

def format_files(frame)->([[float],[float],[int],[int],[int],[int]],[str]):
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
        actual.append(row[13])
    return columns, actual

def normalise(points:[[float],[float],[int],[int],[int],[int]], actuals: [str])->([Weighted_Point,[float,float,float,float,float,float],[float,float,float,float,float,float]]):
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
        if actuals[i] == "w":
            weight = 297/153
        else:
             weight = 297/144
        temp_point = Weighted_Point(this_row1,this_row2,this_row3,this_row4,this_row5,this_row6, weight, classification = actuals[i])
        rows.append(temp_point)
    return rows,mins,maxs
        
def classify_test(points:[Weighted_Point]):
    correct = 0
    incorrect = 0
    for i in points:
        classification = knn_classify(i, points,15)
        if classification == i.classification:
            correct += 1
        else:
            incorrect += 1

if __name__ == "__main__":
    frame = open_files()
    columns, actual = format_files(frame)
    points, mins, maxs = normalise(columns,actual)
    classify_test(points)