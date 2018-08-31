import numpy as np
### HORIZONTAL YARNS = 1
### VERTICAL YARNS = 0

def transform_patch(p):
    p1 = np.swapaxes(p, 0, 2)
    return p1


def create_7s_images(front, back, groundtruth=None):
    if groundtruth is None:
        res = np.concatenate((front, back), axis=2)
    else:
        groundtruth = np.expand_dims(np.uint8(np.round(groundtruth[:, :, 0]/128)), axis=2)
        res = np.concatenate((front, back, groundtruth), axis=2)
    return res