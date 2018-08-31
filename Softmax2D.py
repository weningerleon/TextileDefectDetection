from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding
import numpy as np

class Softmax2D(Layer):
    '''Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __softmax2d(self, x):
        if K.ndim(x) == 4: ### THIS IS ADDED TO NORMAL SOFTMAX
            e = K.exp(x - K.max(x, axis=1, keepdims=True))
            s = K.sum(e, axis=1, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax2d to a tensor '
                             'that is not 4D. '
                             'Here, ndim=' + str(K.ndim(x)))

    def __init__(self, trainable=False, activation=None, **kwargs):
        self.supports_masking = True
        super(Softmax2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.__softmax2d(x)

    def get_config(self):
        config = {'activation': 'Softmax2D'}
        base_config = super(Softmax2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def categorical_accuracy_fcn(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return K.mean(K.equal(K.argmax(y_true, axis=1),
                  K.argmax(y_pred, axis=1)))
