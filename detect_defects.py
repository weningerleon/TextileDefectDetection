import matplotlib.pyplot as plt
import numpy as np
import pickle
from yarn_common_functions import floatPoint, floatPointList
import sys
sys.setrecursionlimit(10000)

def calc_error(X, inner_points, robust_cov, threshold):
    mahal_robust_cov = robust_cov.mahalanobis(X)

    faulty_points = []
    fault_count = 0
    for i, x in enumerate(inner_points):
        if mahal_robust_cov[i] > threshold:
            faulty_points.append(x)
            x.label = "faulty"
            fault_count += 1

    return fault_count, faulty_points


def calculate_density(faulty_points):
    density=0
    max_d=0
    for p1 in faulty_points:
        point_density=0
        for p2 in faulty_points:
            if p1==p2:
                continue
            dist = floatPoint.dist(p1,p2)
            point_density += 1/dist
        if point_density>max_d:
            max_d = point_density
        density+=point_density
    return density, max_d


def detectDefects_warps(file_name, target_name, robust_cov, image=None, write_images=False, threshold=30):
    #robust_cov = pickle.load(open(MCD_path, "rb" ))
    float_points_list = pickle.load(open(file_name, "rb" ))
    for pt in float_points_list:
        pt.faulty=False
    #update distances, since they were changed
    float_points_list.calcDistances()

    x_vals = [fp.x for fp in float_points_list]
    y_vals = [fp.y for fp in float_points_list]

    mean_ver_dist = float_points_list.getMedianWarpDist()

    min_x = min(x_vals) + 0.5*mean_ver_dist
    max_x = max(x_vals) - 0.5*mean_ver_dist
    min_y = min(y_vals) + 1.2*mean_ver_dist
    max_y = max(y_vals) - 1.2*mean_ver_dist

    inner_points = floatPointList()
    for pt in float_points_list:
        if min_x < pt.x < max_x and min_y < pt.y < max_y:
            inner_points.append(pt)



    #TODO: http://scikit-image.org/docs/dev/api/skimage.measure.html
    X = np.zeros([len(inner_points),3])

    for idx, pt in enumerate(inner_points):
        X[idx, 0] = pt.area
        X[idx, 1] = pt.lower_dist
        X[idx, 2] = pt.upper_dist



    fault_count, faulty_points = calc_error(X, inner_points, robust_cov, threshold)

    faults00 = []  # up left
    faults01 = []  # up right
    faults10 = []  # down left
    faults11 = []  # down right

    num_points00 = 0
    num_points01 = 0
    num_points10 = 0
    num_points11 = 0

    mean_x = (max_x + min_x) / 2
    mean_y = (max_y + min_y) / 2
    overlap = 50

    for point in inner_points:
        if point.x < mean_x + overlap and point.y < mean_y + overlap:
            num_points00 += 1
        if point.x > mean_x - overlap and point.y < mean_y + overlap:
            num_points01 += 1
        if point.x < mean_x + overlap and point.y > mean_y - overlap:
            num_points10 += 1
        if point.x > mean_x - overlap and point.y > mean_y - overlap:
            num_points11 += 1

    for point in faulty_points:
        if point.x < mean_x + overlap and point.y < mean_y + overlap:
            faults00.append(point)
        if point.x > mean_x - overlap and point.y < mean_y + overlap:
            faults01.append(point)
        if point.x < mean_x + overlap and point.y > mean_y - overlap:
            faults10.append(point)
        if point.x > mean_x - overlap and point.y > mean_y - overlap:
            faults11.append(point)

    # num_inner_points = inner_points.__len__()
    # density = fault_count / num_inner_points * 100
    # density, max_d = calculate_density(faulty_points)
    density00 = faults00.__len__() / num_points00 * 100
    density01 = faults01.__len__() / num_points01 * 100
    density10 = faults10.__len__() / num_points10 * 100
    density11 = faults11.__len__() / num_points11 * 100

    #print(file_name + " fault count von: " + str(fault_count))
    #print(file_name + " Faults per 100 flt points: " + str(density))

    if write_images==True:
        float_points_list.showPoints(image)
        plt.savefig(target_name + '.png')
        plt.close("all")
    #return fault_count
    return density00, density01, density10, density11

def detectDefects_wefts(file_name, target_name, robust_cov, image=None, write_images=False, threshold=30):
    #robust_cov = pickle.load(open(MCD_path, "rb" ))
    float_points_list = pickle.load(open(file_name, "rb" ))
    for pt in float_points_list:
        pt.faulty=False
    #update distances, since they were changed
    float_points_list.calcDistances()

    x_vals = [fp.x for fp in float_points_list]
    y_vals = [fp.y for fp in float_points_list]

    mean_hor_dist = float_points_list.getMedianWeftDist()

    min_x = min(x_vals) + 1.2*mean_hor_dist
    max_x = max(x_vals) - 1.2*mean_hor_dist
    min_y = min(y_vals) + .5*mean_hor_dist
    max_y = max(y_vals) - .5*mean_hor_dist

    inner_points = floatPointList()
    for pt in float_points_list:
        if min_x < pt.x < max_x and min_y < pt.y < max_y:
            inner_points.append(pt)



    #TODO: http://scikit-image.org/docs/dev/api/skimage.measure.html
    X = np.zeros([len(inner_points),3])

    for idx, pt in enumerate(inner_points):
        X[idx,0] = pt.area
        X[idx,1] = pt.right_dist
        X[idx,2] = pt.left_dist


    fault_count, faulty_points = calc_error(X, inner_points, robust_cov, threshold)

    faults00 = []    #up left
    faults01 = []   #up right
    faults10 = []   #down left
    faults11 = []   #down right

    num_points00 = 0
    num_points01 = 0
    num_points10 = 0
    num_points11 = 0

    mean_x = (max_x + min_x) / 2
    mean_y = (max_y + min_y) / 2
    overlap = 50

    for point in inner_points:
        if point.x < mean_x+overlap and point.y < mean_y+overlap:
            num_points00 += 1
        if point.x > mean_x - overlap and point.y < mean_y + overlap:
            num_points01 += 1
        if point.x < mean_x + overlap and point.y > mean_y - overlap:
            num_points10 += 1
        if point.x > mean_x - overlap and point.y > mean_y - overlap:
            num_points11 += 1


    for point in faulty_points:
        if point.x < mean_x+overlap and point.y < mean_y+overlap:
            faults00.append(point)
        if point.x > mean_x - overlap and point.y < mean_y + overlap:
            faults01.append(point)
        if point.x < mean_x + overlap and point.y > mean_y - overlap:
            faults10.append(point)
        if point.x > mean_x - overlap and point.y > mean_y - overlap:
            faults11.append(point)

    #num_inner_points = inner_points.__len__()
    #density = fault_count / num_inner_points * 100
    # density, max_d = calculate_density(faulty_points)
    density00 = faults00.__len__() / num_points00 * 100
    density01 = faults01.__len__() / num_points01 * 100
    density10 = faults10.__len__() / num_points10 * 100
    density11 = faults11.__len__() / num_points11 * 100

    # print(file_name + " fault count von: " + str(fault_count))
    #print(file_name + " Faults per 100 flt points: " + str(density))

    if write_images==True:
        float_points_list.showPoints(image)
        plt.savefig(target_name + '.png')
        plt.close("all")
    return density00, density01, density10, density11
    #return fault_count




