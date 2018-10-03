from tensorflow.keras.layers import Input, concatenate, Dropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization, Softmax

import tensorflow as tf


def createUnet_modified():

    alpha = 0.01
    inputs = Input(shape=(None, None, 6))

    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = LeakyReLU(alpha=alpha)(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(256, kernel_size=(3, 3), padding='same')(up1)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = LeakyReLU(alpha=alpha)(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = LeakyReLU(alpha=alpha)(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, kernel_size=(3, 3), padding='same')(up2)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, kernel_size=(3, 3), padding='same')(up3)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)
    conv7 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)

    out = Conv2D(3, kernel_size=(1, 1), padding='same')(conv7)
    out = Softmax(axis=-1)(out)

    model = tf.keras.models.Model(inputs=inputs, outputs=out)

    return model


if __name__ == "__main__":
    model = createUnet_modified()