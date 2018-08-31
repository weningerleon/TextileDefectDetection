import matplotlib.pyplot as plt
import numpy as np
import pickle
import operator
from yarn_common_functions import floatPoint, floatPointList

import sys
sys.setrecursionlimit(10000)


def check_consistency_weft(weft_points_list):
    for float_point in weft_points_list:
        #check right neigbor
        if float_point.right_n != None and float_point.right_n.left_n != float_point:
            print("wrong weft connection")
            if float_point.right_n.left_n != None and float_point.right_n.left_n.right_n != float_point.right_n:
                #The right neigbor is also not ok
                float_point.right_n.left_n = None
            float_point.right_n = None

        # check left neighbor
        if float_point.left_n != None and float_point.left_n.right_n != float_point:
            print("wrong weft connection")
            if float_point.left_n.right_n != None and float_point.left_n.right_n.left_n != float_point.left_n:
                # The right neigbor is also not ok
                float_point.left_n.right_n = None
            float_point.left_n = None


def check_consistency_warp(warp_points_list):
    for float_point in warp_points_list:

        # check lower neighbor
        if float_point.lower_n != None and float_point.lower_n.upper_n != float_point:
            print("wrong warp connection")
            if float_point.lower_n.upper_n != None and float_point.lower_n.upper_n.lower_n != float_point.lower_n:
                #The right neigbor is also not ok
                float_point.lower_n.upper_n = None
            float_point.lower_n = None

        # check upper neighbor
        if float_point.upper_n != None and float_point.upper_n.lower_n != float_point:
            print("wrong warp connection")
            if float_point.upper_n.lower_n != None and float_point.upper_n.lower_n.upper_n != float_point.upper_n:
                # The right neigbor is also not ok
                float_point.upper_n.lower_n = None
            float_point.upper_n = None


def searchPoint(float_points_list, x, y, radius):
    dists = [np.sqrt(np.square(fp.x - x) + np.square(fp.y - y)) for fp in float_points_list]

    min_index, min_value = min(enumerate(dists), key=operator.itemgetter(1))

    if min_value < radius:
        return float_points_list[min_index]
    return None

def patch_single_warp_connections(warp_points_list, mean_ver_dist):
    search_radius = mean_ver_dist*0.3

    for fp in warp_points_list:
        if fp.lower_n is None:
            res = searchPoint(float_points_list=warp_points_list, x=fp.x, y=fp.y + mean_ver_dist, radius=search_radius)
            if res is not None:
                if res.upper_n is None or res.upper_dist > floatPoint.dist(res, fp):
                    fp.lower_n = res
                    if res.upper_n is not None: res.upper_n.lower_n = None
                    res.upper_n = fp

        if fp.upper_n is None:
            res = searchPoint(float_points_list=warp_points_list, x=fp.x, y=fp.y - mean_ver_dist, radius=search_radius)
            if res != None:
                if res.lower_n is None or res.lower_dist > floatPoint.dist(res, fp):
                    fp.upper_n = res
                    if res.lower_n is not None: res.lower_n.upper_n = None
                    res.lower_n = fp



def patch_single_weft_connections(weft_points_list, mean_hor_dist):
    search_radius = mean_hor_dist*0.3

    for fp in weft_points_list:
        if fp.right_n is None:
            res = searchPoint(float_points_list=weft_points_list, x=fp.x + mean_hor_dist, y=fp.y, radius=search_radius)
            if res is not None:
                if res.left_n is None or res.left_dist > floatPoint.dist(res,fp):
                    fp.right_n = res
                    if res.left_n is not None: res.left_n.right_n = None
                    res.left_n = fp

        if fp.left_n is None:
            res = searchPoint(float_points_list=weft_points_list, x=fp.x - mean_hor_dist, y=fp.y, radius=search_radius)
            if res is not None:
                if res.right_n is None or res.right_dist > floatPoint.dist(res, fp):
                    fp.left_n = res
                    if res.right_n is not None: res.right_n.left_n = None
                    res.right_n = fp


def patchDoubleWarps(warp_points_list, mean_ver_dist):
    min_dist = mean_ver_dist * 0.5

    for fl in warp_points_list:
        if fl.lower_n is not None:
            if floatPoint.dist(fl, fl.lower_n) < min_dist:
                print("double warp found")
                # They belong together
                temp = fl.lower_n.lower_n
                fl.x = (fl.x + fl.lower_n.x) / 2
                fl.y = (fl.y + fl.lower_n.y) / 2
                fl.area = fl.area + fl.lower_n.area #can be made better!
                warp_points_list.remove(fl.lower_n)
                fl.lower_n = temp


def patchDoubleWefts(weft_points_list, mean_hor_dist):
    min_dist = mean_hor_dist * 0.5

    for fl in weft_points_list:
        if fl.right_n is not None:
            if floatPoint.dist(fl, fl.right_n) < min_dist:
                print("double weft found")

                # They belong together
                temp = fl.right_n.right_n
                fl.x = (fl.x + fl.right_n.x) / 2
                fl.y = (fl.y + fl.right_n.y) / 2
                fl.area = fl.area + fl.right_n.area
                weft_points_list.remove(fl.right_n)
                fl.right_n = temp



def correctYarns(file_name_weft, file_name_warp, target_name, image=None, write_images=False):
    weft_points_list = pickle.load(open(file_name_weft, "rb" ))
    warp_points_list = pickle.load(open(file_name_warp, "rb" ))

    check_consistency_warp(warp_points_list)
    check_consistency_weft(weft_points_list)

    mean_ver_dist = warp_points_list.getMedianWarpDist()
    mean_hor_dist = weft_points_list.getMedianWeftDist()

    patch_single_warp_connections(warp_points_list, mean_ver_dist)
    patch_single_weft_connections(weft_points_list, mean_hor_dist,)

    patchDoubleWarps(warp_points_list, mean_ver_dist)
    patchDoubleWefts(weft_points_list, mean_hor_dist)


    float_points_list = floatPointList(warp_points_list + weft_points_list)
    float_points_list.calcDistances()

    if write_images==True:
        float_points_list.showPoints(image)
        plt.savefig(target_name + '.png')
        plt.close("all")
    pickle.dump(warp_points_list, open(target_name + '__warp.p', "wb"))
    pickle.dump(weft_points_list, open(target_name + '__weft.p', "wb"))
