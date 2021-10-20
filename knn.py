import pandas as pd
import glob
import os
import math

class Point(object):
    def __init__(self, coord1,coord2,coord3,coord4,coord5,coord6, classification=None, distance=None):
        self.dimension1 = coord1
        self.dimension2 = coord2
        self.dimension3 = coord3
        self.dimension4 = coord4
        self.dimension5 = coord5
        self.dimension6 = coord6
        self.classification = classification
        self.distance = distance

    def distance_to(self, point2)->float:
        distance1_squared = (self.dimension1-point2.dimension2)**2+(self.dimension2-point2.dimension2)**2
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


def knn_classify(point:Point, classified_points:Point, k:int)->str:
    best_points = []
    for classified_point in classified_points:
        classified_point.set_distance(point)
        if len(best_points) < k:
            best_points.append(classified_point)
            if len(best_points) == k:
                best_points.sort()
                best_points.reverse()
        elif classified_point < best_points[0]:
            i = 0
            while classified_point < best_points[i] and i != k:
                i += 1
            if i == 0:
                best_points.pop(0)
                best_points.insert(0, classified_point)
            else:
                best_points.insert(i, classified_point)
                best_points.pop(0)
    w_count = 0
    l_count = 0
    for best in best_points:
        if best.classification == "w":
            w_count += 1
        else: 
            l_count += 1
    if w_count > l_count:
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
    return frame

if __name__ == "__main__":
    test_point = Point(2,2,2,2,2,2)
    point1 = Point(0,0,0,0,0,0,"l")
    point2 = Point(1,1,1,1,1,1,"l")
    point3 = Point(3,3,3,3,3,3,"w")
    point4 = Point(3,3,3,3,3,3,"w")
    print(knn_classify(test_point,[point1,point2,point3,point4],3))