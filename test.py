import cv2
import numpy as np
import itertools
import time
import tensorflow as tf
from keras import models

### HORIZONTAL YARNS = 1
### VERTICAL YARS = 0


def transform_patch(p):
    p1 = np.rollaxis(p, axis=2)
    return p1

def to_img_fast(data):
    res = np.rint(data[0,1,:,:])*127 + np.rint(data[0,2,:,:])*255
    return np.uint8(res)

def test(model, img, target_name):
    shape = img.shape
    img=img[:int(shape[0]/8)*8,:int(shape[1]/8)*8,:]
    max_length = 920
    overlap = 40
    max_length = max_length - 2*overlap
    img = np.lib.pad(img, ((overlap, overlap),(overlap, overlap), (0,0)), 'symmetric')
    img = transform_patch(img)
    img = img / 255
    data = np.expand_dims(img, axis=0)

    data_arr = np.zeros(shape=(1,3,data.shape[2]-2*overlap,data.shape[3]-2*overlap))
    for x,y in itertools.product(range(overlap,data.shape[2],max_length),range(overlap,data.shape[3],max_length)):
         temp2 = model.predict(data[:,:,x-overlap:x+max_length+overlap,y-overlap:y+max_length+overlap])
         temp = temp2[:,:,overlap:-overlap,overlap:-overlap]
         data_arr[:, :, x-overlap:x + temp.shape[2]-overlap, y-overlap:y + temp.shape[3]-overlap] = temp

    #start = time.time()
    res = to_img_fast(data_arr)
    #end = time.time()
    #print(end - start)

    cv2.imwrite(target_name, res)
