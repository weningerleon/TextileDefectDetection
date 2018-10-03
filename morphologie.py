import cv2
import numpy as np
import glob
import os
from pathlib import WindowsPath

from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, binary_fill_holes


def morphologie(img_name, target_dir, target_name):
    img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)

    thresh_hor = 195
    thresh_ver = 60
    hor = cv2.threshold(img, thresh_hor, 255, cv2.THRESH_BINARY)[1]
    ver = 255-cv2.threshold(img, thresh_ver, 255, cv2.THRESH_BINARY)[1]

    mat = np.ones((5,5))
    hor = binary_opening(hor, structure=mat, iterations=2).astype(np.uint8) * 255
    #hor = binary_closing(hor, structure=mat, iterations=1).astype(np.uint8)*255

    #mat = np.ones((3,3))
    ver = binary_opening(ver, structure=mat, iterations=2).astype(np.uint8) * 255
    #ver = binary_closing(ver, structure=mat, iterations=1).astype(np.uint8)*255

    cv2.imwrite(os.path.join(target_dir, 'h' + target_name + '.png'), hor)
    cv2.imwrite(os.path.join(target_dir, 'v' + target_name + '.png'), ver)
