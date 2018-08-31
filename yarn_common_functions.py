import matplotlib.pyplot as plt
import numpy as np
from statistics import median
import cv2

class floatPoint:
    x=0
    y=0
    arr_x = -1
    arr_y = -1
    right_n = None
    right_dist = -1
    left_n = None
    left_dist = -1
    upper_n = None
    upper_dist = -1
    lower_n = None
    lower_dist = -1

    area = -1
    convArea = -1
    label = ""

    def __init__(self,x,y,label, area, convArea):
        self.x = x
        self.y = y
        self.label = label
        self.area = area
        self.convArea = convArea

    @staticmethod
    def dist_x(p1, p2):
        return np.absolute(p1.x - p2.x)

    @staticmethod
    def dist_y(p1, p2):
        return np.absolute(p1.y - p2.y)

    @staticmethod
    def dist(p1, p2):
        return np.sqrt(np.square(p1.x - p2.x) + np.square(p1.y - p2.y))

    def find_neighbors(self, all_points):
        myX = self.x
        myY = self.y
        # ax.plot(myX, myY, '.r', markersize=15)
        tube_radius = np.sqrt(self.area) / 4

        no_x = False
        try:
            x_dists, x_points = zip(
                *[(point.x - myX, point) for point in all_points if
                  (myY - tube_radius < point.y < myY + tube_radius and point.x != myX)])
            x_dists = list(x_dists)
        except Exception as inst:
            no_x = True

        try:
            y_dists, y_points = zip(
                *[(point.y - myY, point) for point in all_points if
                  (myX - tube_radius < point.x < myX + tube_radius and point.y != myY)])
            y_dists = list(y_dists)
        except Exception as inst:
            no_y = True
            if no_x == True:
                return False

                # right neighbor
        try:
            arg = min([x for x in x_dists if x > 0])
            idx = x_dists.index(arg)
            self.right_n = x_points[idx]
        except Exception as inst:
            self.right_n = None
        '''    '''
        # left neighbor
        try:
            arg = max([x for x in x_dists if x < 0])
            idx = x_dists.index(arg)
            self.left_n = x_points[idx]
        except Exception as inst:
            self.left_n = None

        # upper neighbor
        try:
            arg = max([y for y in y_dists if y < 0])
            idx = y_dists.index(arg)
            self.upper_n = y_points[idx]
        except Exception as inst:
            self.upper_n = None

        # lower neighbor
        try:
            arg = min([y for y in y_dists if y > 0])
            idx = y_dists.index(arg)
            self.lower_n = y_points[idx]
        except Exception as inst:
            self.lower_n = None

        return True

    def find_warp_neighbors(self, all_points):
        myX = self.x
        myY = self.y
        # ax.plot(myX, myY, '.r', markersize=15)
        tube_radius = np.sqrt(self.area) / 4

        try:
            y_dists, y_points = zip(
                *[(point.y - myY, point) for point in all_points if
                  (myX - tube_radius < point.x < myX + tube_radius and point.y != myY)])
            y_dists = list(y_dists)
        except Exception as inst:
            return False

        # upper neighbor
        try:
            arg = max([y for y in y_dists if y < 0])
            idx = y_dists.index(arg)
            self.upper_n = y_points[idx]
        except Exception as inst:
            self.upper_n = None

        # lower neighbor
        try:
            arg = min([y for y in y_dists if y > 0])
            idx = y_dists.index(arg)
            self.lower_n = y_points[idx]
        except Exception as inst:
            self.lower_n = None

        return True

    def find_weft_neighbors(self, all_points):
        myX = self.x
        myY = self.y
        # ax.plot(myX, myY, '.r', markersize=15)
        tube_radius = np.sqrt(self.area) / 4

        try:
            x_dists, x_points = zip(
                *[(point.x - myX, point) for point in all_points if
                  (myY - tube_radius < point.y < myY + tube_radius and point.x != myX)])
            x_dists = list(x_dists)
        except Exception as inst:
            return False

            # right neighbor
        try:
            arg = min([x for x in x_dists if x > 0])
            idx = x_dists.index(arg)
            self.right_n = x_points[idx]
        except Exception as inst:
            self.right_n = None
        '''    '''
        # left neighbor
        try:
            arg = max([x for x in x_dists if x < 0])
            idx = x_dists.index(arg)
            self.left_n = x_points[idx]
        except Exception as inst:
            self.left_n = None

        return True





class floatPointList(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def remove(self, fp):
        if fp.lower_n is not None:
            fp.lower_n.upper_n = None
            if fp.right_n is not None:
                fp.right_n.left_n = None
            if fp.upper_n is not None:
                fp.upper_n.lower_n = None
            if fp.left_n is not None:
                fp.left_n.right_n = None

        list.remove(self, fp)

    def getMedianDists(self):
        means_hor = []
        means_ver = []
        for fl in self:
            if fl.right_dist > 0:
                means_hor.append(fl.right_dist)
            if fl.lower_dist > 0:
                means_ver.append(fl.lower_dist)

        mean_hor = median(means_hor)
        mean_ver = median(means_ver)
        return mean_hor, mean_ver

    def getMedianWeftDist(self):
        means_ver = []
        for fl in self:
            if fl.right_dist > 0:
                means_ver.append(fl.right_dist)

        mean_ver = median(means_ver)
        return mean_ver

    def getMedianWarpDist(self):
        means_ver = []
        for fl in self:
            if fl.lower_dist > 0:
                means_ver.append(fl.lower_dist)

        mean_ver = median(means_ver)
        return mean_ver

    def showPoints(self, image=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        axes = plt.gca()
        axes.invert_yaxis()

        if image is not None:
            plt.axis('on')
            gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray_im)
            plt.imshow(cl1, cmap='gray')
            axes.set_xlim(left=0, right=1900)
            axes.set_ylim(top=0, bottom=1900)

        for myFloatPoint in self:
            if myFloatPoint.label == "warp_float":
                ax.plot(myFloatPoint.x, myFloatPoint.y, '.b', markersize=10)
            elif myFloatPoint.label == "weft_float":
                ax.plot(myFloatPoint.x, myFloatPoint.y, '.g', markersize=10)
            else:
                ax.plot(myFloatPoint.x, myFloatPoint.y, '.r', markersize=10)
            x1 = myFloatPoint.x
            y1 = myFloatPoint.y

            if myFloatPoint.lower_n:
                x2 = myFloatPoint.lower_n.x
                y2 = myFloatPoint.lower_n.y
                plt.plot([x1, x2], [y1, y2], 'r')
            if myFloatPoint.right_n:
                x2 = myFloatPoint.right_n.x
                y2 = myFloatPoint.right_n.y
                plt.plot([x1, x2], [y1, y2], 'g')




    def calcDistances(self):
        for point in self:
            if point.right_n:
                dist = floatPoint.dist_x(point, point.right_n)
                point.right_dist = dist
                point.right_n.left_dist = dist
            else:
                point.right_dist = -1
            if point.lower_n:
                dist = floatPoint.dist_y(point, point.lower_n)
                point.lower_dist = dist
                point.lower_n.upper_dist = dist
            else:
                point.lower_dist = -1


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y