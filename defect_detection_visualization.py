import cv2

from detect_defects import calc_error
import matplotlib.image as mpimg
from stats_of_cloth import *
from scipy.stats import gaussian_kde


def paintDefects_warps(file_name, robust_cov, threshold=30):
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

    X = np.zeros([len(inner_points),3])

    for idx, pt in enumerate(inner_points):
        X[idx, 0] = pt.area
        X[idx, 1] = pt.lower_dist
        X[idx, 2] = pt.upper_dist


    fault_count, faulty_points = calc_error(X, inner_points, robust_cov, threshold)



    x = np.zeros(faulty_points.__len__())
    y = np.zeros(faulty_points.__len__())
    for idx, myFloatPoint in enumerate(faulty_points):
        x[idx] = myFloatPoint.x
        y[idx] = myFloatPoint.y
        #ax.plot(myFloatPoint.x, myFloatPoint.y, color='red', marker='s', markersize=30, alpha=0.15)

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, color='red')# c=z, marker='s', s=500, edgecolor='', alpha=0.5)
    x=3

def paintDefects_wefts(file_name, robust_cov, threshold=30):
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

    X = np.zeros([len(inner_points),3])

    for idx, pt in enumerate(inner_points):
        X[idx,0] = pt.area
        X[idx,1] = pt.right_dist
        X[idx,2] = pt.left_dist


    fault_count, faulty_points = calc_error(X, inner_points, robust_cov, threshold)



    x = np.zeros(faulty_points.__len__())
    y = np.zeros(faulty_points.__len__())
    for idx, myFloatPoint in enumerate(faulty_points):
        x[idx] = myFloatPoint.x
        y[idx] = myFloatPoint.y
        #ax.plot(myFloatPoint.x, myFloatPoint.y, color='red', marker='s', markersize=30, alpha=0.15)

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, color='red')# c=z, marker='s', s=500, edgecolor='', alpha=0.5)
    #plt.show()
    x=3



#######################################################################################################################
#######################################################################################################################
threshold=30

warp_path = 'V:/ITA/MA_weninger/plain_weave/Fabric1/yarns_corrected/c0130c___warp.p'
weft_path = 'V:/ITA/MA_weninger/plain_weave/Fabric1/yarns_corrected/c0130c___weft.p'

image = mpimg.imread('V:/ITA/MA_weninger/plain_weave/Fabric1/test_images/fl0130c.png')

robust_cov_warp = pickle.load(open('V:/ITA/MA_weninger/plain_weave/Fabric1/MCD_warp.p', "rb"))
robust_cov_weft = pickle.load(open('V:/ITA/MA_weninger/plain_weave/Fabric1/MCD_weft.p', "rb"))

target_name = 'Y:\temp\test'

fig, ax = plt.subplots(figsize=(10, 10))
axes = plt.gca()
axes.invert_yaxis()

if image is not None:
    plt.axis('off')
    plt.imshow(image)
    axes.set_xlim(left=0, right=1900)
    axes.set_ylim(top=0, bottom=1900)

paintDefects_wefts(file_name=weft_path, robust_cov=robust_cov_weft, threshold=threshold)
paintDefects_warps(file_name=warp_path, robust_cov=robust_cov_warp, threshold=threshold)
plt.show()
x=3
#        dens00e, density01e, dens10e, dens11e = detectDefects_wefts(file_name=weft_path, target_name=target_name_weft,
#                                                                    robust_cov=robust_cov_weft, write_images=False,
#                                                                    image=fl_im, threshold=threshold1)

