import matplotlib.pyplot as plt
import cv2
import pickle


from yarn_common_functions import floatPoint, floatPointList, line_intersection
from skimage.measure import label, regionprops


import sys
sys.setrecursionlimit(10000)



def detectYarns(hor_name, ver_name, target_name, image=None, write_images=False):
    hor = cv2.imread(hor_name, cv2.IMREAD_GRAYSCALE)
    ver = cv2.imread(ver_name, cv2.IMREAD_GRAYSCALE)

    labels_hor = label(hor)
    regions_hor = regionprops(labels_hor)

    labels_ver = label(ver)
    regions_ver = regionprops(labels_ver)


    warp_points_list = floatPointList()
    weft_points_list = floatPointList()

    #Die float-points sind um pwh pixel verschoben, da das erkannte Bild um genau diese Anzahl Pixel kleiner ist als das original
    #Nicht bei fcn!!
    for props in regions_hor:
        y0, x0 = props.centroid
        temp = floatPoint(x=x0,y=y0,label="weft_float", area=props.area, convArea=props.convex_area)
        weft_points_list.append(temp)

    for props in regions_ver:
        y0, x0 = props.centroid
        temp = floatPoint(x=x0,y=y0,label="warp_float", area=props.area, convArea=props.convex_area)
        warp_points_list.append(temp)


    for myFloatPoint in warp_points_list:
        myFloatPoint.find_warp_neighbors(warp_points_list)

    for myFloatPoint in weft_points_list:
        myFloatPoint.find_weft_neighbors(weft_points_list)

    weft_points_list.calcDistances()
    warp_points_list.calcDistances()
    pickle.dump(weft_points_list,  open(target_name + '__weft.p', "wb" ))
    pickle.dump(warp_points_list,  open(target_name + '__warp.p', "wb" ))

    float_points_list = floatPointList(warp_points_list + weft_points_list)
    if write_images==True:
        float_points_list.showPoints(image)
        plt.savefig(target_name + '.png', bbox_inches='tight', pad_inches=0)
        plt.close("all")



