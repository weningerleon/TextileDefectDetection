from keras.models import Model, Sequential
from keras.layers import Input, merge, Dropout

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from Softmax2D import Softmax2D


def createUnet_modified():

    alpha = 0.01
    inputs = Input((6, None, None))
    conv1 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(inputs)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, border_mode='same', init='he_normal')(pool3)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Convolution2D(512, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    #
    up1 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv5 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(up1)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = LeakyReLU(alpha=alpha)(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = LeakyReLU(alpha=alpha)(conv5)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv2], mode='concat', concat_axis=1)
    conv6 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(up2)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)
    #
    up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv1], mode='concat', concat_axis=1)
    conv7 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up3)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)
    conv7 = Convolution2D(3, 3, 3, border_mode='same', init='he_uniform')(conv7)
    ############

    outputs = Softmax2D()(conv7)

    model = Model(input=inputs, output=outputs)

    return model




def createFCN():
    model = Sequential()
    index = 0
    alpha = 0.01
    kernel = 3
    filter_size = 64
    pool_size = 2

    model.add(Convolution2D(64, 3, 3, input_shape=(6, None, None), border_mode='same', init='he_uniform'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_uniform'))  # => 64x128x128
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))  # => 64x16x16

    # Second Convolution
    model.add(Convolution2D(192, 3, 3, border_mode='same', init='he_uniform'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Convolution2D(192, 3, 3, border_mode='same', init='he_uniform'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Convolution2D(192, 3, 3, border_mode='same', init='he_uniform'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully Convolutional Layers
    model.add(Convolution2D(384, 5, 5, border_mode='same', init='he_uniform'))  # => 256x16x16
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.5))
    model.add(Convolution2D(384, 5, 5, border_mode='same', init='he_uniform'))  # => 256x16x16
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    #model.add(Dropout(0.5))

    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', init='he_uniform'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Convolution2D(3, 3, 3, border_mode='same', init='he_uniform'))
    model.add(Softmax2D())  # Attention, softmax2d self implemented
    return model


def createUnet_simple():
    #conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    #conv1 = LeakyReLU(alpha=alpha)(conv1)
    inputs = Input((6, None, None))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    conv6 = Convolution2D(3, 1, 1, activation='relu', border_mode='same')(conv5)
    ############

    outputs = Softmax2D()(conv6)

    model = Model(input=inputs, output=outputs)

    return model