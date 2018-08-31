import pickle
from yarn_common_functions import floatPointList
import numpy as np
from sklearn.covariance import MinCovDet

def compute_MCD_weft(weftsPickled, target_path):

    weft_points_list = floatPointList()
    for pickled_path in weftsPickled:
        weft_points_list.extend(pickle.load(open(pickled_path, "rb" )))

    x_vals = [fp.x for fp in weft_points_list]
    y_vals = [fp.y for fp in weft_points_list]

    mean_hor_dist = weft_points_list.getMedianWeftDist()

    min_x = min(x_vals) + 1.5 * mean_hor_dist
    max_x = max(x_vals) - 1.5 * mean_hor_dist
    min_y = min(y_vals) + 1.5 * mean_hor_dist
    max_y = max(y_vals) - 1.5 * mean_hor_dist

    inner_points = floatPointList()
    for pt in weft_points_list:
        if min_x < pt.x < max_x and min_y < pt.y < max_y:
            inner_points.append(pt)

    X = np.zeros([len(inner_points), 3])

    for idx, pt in enumerate(inner_points):
        X[idx,0] = pt.area
        X[idx,1] = pt.right_dist
        X[idx,2] = pt.left_dist

    Y = X[~(X<=0).any(axis=1)]

    robust_cov = MinCovDet(support_fraction=0.8).fit(Y)
    pickle.dump(robust_cov, open(target_path, "wb"))

def compute_MCD_warp(warpsPickled, target_path):

    warp_points_list = floatPointList()
    for pickled_path in warpsPickled:
        warp_points_list.extend(pickle.load(open(pickled_path, "rb" )))

    x_vals = [fp.x for fp in warp_points_list]
    y_vals = [fp.y for fp in warp_points_list]

    mean_ver_dist = warp_points_list.getMedianWarpDist()

    min_x = min(x_vals) + 1.5 * mean_ver_dist
    max_x = max(x_vals) - 1.5 * mean_ver_dist
    min_y = min(y_vals) + 1.5 * mean_ver_dist
    max_y = max(y_vals) - 1.5 * mean_ver_dist

    inner_points = floatPointList()
    for pt in warp_points_list:
        if min_x < pt.x < max_x and min_y < pt.y < max_y:
            inner_points.append(pt)


    #####CHANGED
    #print("attention, only 2D!!!!!")
    X = np.zeros([len(inner_points), 3])

    for idx, pt in enumerate(inner_points):
        X[idx,0] = pt.area
        X[idx,1] = pt.lower_dist
        X[idx,2] = pt.upper_dist


    Y = X[~(X<=0).any(axis=1)]

    robust_cov = MinCovDet(support_fraction=0.8).fit(Y)
    pickle.dump(robust_cov, open(target_path, "wb"))
