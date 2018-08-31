from __future__ import print_function

import keras
import keras.metrics
import pickle
import math
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import itertools
import cv2
import tensorflow as tf
tf.python.control_flow_ops = tf
from Softmax2D import categorical_accuracy_fcn
from createNetwork import *

def to_categorical3D(data):
    num_categories=np.int(np.max(data)+1)

    res = np.zeros([data.shape[0], data.shape[1],num_categories])
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            res[x, y, data[x, y]] = 1

    return res


def transform_patch(p):
    if np.ndim(p)==3:
        p1 = np.rollaxis(p, axis=2)
    elif np.ndim(p)==2:
        p1 = np.expand_dims(p,axis=0)
    else:
        print("error in transform_patch")
    return p1


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def batch_generator_augmented(imgs, size_patch=(352,400), overlap=(2,2)):
    # The size needs to be divisible by 4,8, or 16 (according to the network!!!
    # Comment/De-comment the wished augmentation techniques
    #data_augmentation: turn +-3°, stretch
    #Bekommt als input Images mit 7 Kanaelen. Davon Kanäle 0-2: Auflicht, 3-5: Durchlicht, 6: Groundtruth

    X_train = []
    Y_train = []

    stretch_factors = [0.8, 1, 1.2]
    rotate_factors = [-2,0,2]

    for idx, img in enumerate(imgs):
        print("image: " + str(idx), "shape:" + str(img.shape))
        count = 0
        #for slice1, slice2, slice3 in itertools.product((0,1),(1,2),(0,2)):
        #for slice1, slice2, slice3 in itertools.product((0), (1), (2)):
        #print(" " + str(slice1) + str(slice2) + str(slice3))
        for stfx, stfy, rtf in itertools.product(stretch_factors, stretch_factors, rotate_factors):
            for x in range(0,int((img.shape[0]-size_patch[0])*stfx),size_patch[0]-overlap[0]):
                for y in range(0,int((img.shape[1]-size_patch[1]*stfy)),size_patch[1]-overlap[1]):
                    #print("x: " + str(x) + ", y: " + str(y) + ", stfx: " + str(stfx) + ", stfy: " + str(stfy) + ", rtf: " + str(rtf))
                    count+=1
                    patch = img[x:int(x+size_patch[0]*stfx), y:int(y+size_patch[1]*stfy)] # augmentation by scaling

                    # augmentation by rotation
                    M = cv2.getRotationMatrix2D((patch.shape[1] / 2, patch.shape[0] / 2), rtf, 1)
                    dst = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))
                    image_rotated_cropped = crop_around_center(dst,*largest_rotated_rect(patch.shape[1], patch.shape[0],rtf*np.pi/180))


                    patch = cv2.resize(src=image_rotated_cropped, dsize=size_patch)

                    X_patch = patch[:,:,0:6]

                    # augmentation by color
                    #X_patch[:, :, 0], X_patch[:, :, 1], X_patch[:, :, 2] = patch[:, :, slice1], patch[:, :, slice2], patch[:, :, slice3]

                    Y_patch = patch[:,:,6]

                    Y_patch = to_categorical3D(Y_patch)
                    X_patch = transform_patch(X_patch)
                    Y_patch = transform_patch(Y_patch)

                    X_train.append(X_patch)
                    Y_train.append(Y_patch)
        print(str(count) + " training images generated")
    print("Converting to array 1")
    X_train = np.asarray(X_train).astype('float32')
    print("Converting to array 2")
    Y_train = np.asarray(Y_train).astype('float32')
    print("Finished converting")

    X_train /= 255

    return X_train,Y_train


class AccLogger(Callback):
    def on_train_begin(self, logs={}):
        self.nbatch = 20
        self.num = 0
        self.accs = []
        #self.val_accs = []
        self.loss = []
        self.epoch_ends = []

    def on_batch_end(self, batch, logs={}):
        self.num +=1
        if self.num % self.nbatch == 0:
            self.accs.append(logs.get('categorical_accuracy_fcn'))
            self.loss.append(logs.get('loss'))
            self.num = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_ends.append(self.accs.__len__())
        #self.val_accs.append(logs.get('val_categorical_accuracy_fcn'))

def train(model_name,imgs):
    #Function for trainign of the networks

    model = createUnet_modified()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy_fcn])


    X_train, Y_train = batch_generator_augmented(imgs)

    modelcheckpoint = ModelCheckpoint(model_name, monitor='loss', verbose=0, save_best_only=False,
                                      save_weights_only=False, mode='auto', period=1)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

    acclogger = AccLogger()

    print("training")

    model.fit(X_train, Y_train, batch_size=2, nb_epoch=5, validation_split=0.0, shuffle=True,
              callbacks=[acclogger, modelcheckpoint])  # , class_weight={0.33,0.33,0.33}

    pickle.dump((acclogger.accs,acclogger.loss,acclogger.epoch_ends), open(model_name[:-3] + "_complete_logger.p", "wb"))

    print("training finished")
    model.save(model_name)

def retrain(model_name,imgs):
    #This function can be used to re-train a network on some data.
    #e.g. train on all plain weave image, then retrain on a twill weave fabric

    model = keras.models.load_model(model_name, custom_objects={"Softmax2D": Softmax2D, "categorical_accuracy_fcn": categorical_accuracy_fcn})

    X_train, Y_train = batch_generator_augmented(imgs)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    print("retraining")

    model.fit(X_train, Y_train, batch_size=2, nb_epoch=1, validation_split=0.0, shuffle=True)  # , class_weight={0.33,0.33,0.33}

    print("retraining finished")

    model.save(model_name)
    print("saved")