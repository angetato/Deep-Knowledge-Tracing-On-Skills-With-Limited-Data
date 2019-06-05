# coding : utf-8

# Goal: Text classification using word embedding - RNN  + Attention layer
# source: https://www.kaggle.com/artgor/eda-and-lstm-cnn

# Source: https://github.com/Diyago/ML-DL-scripts/blob/9e161a96580efa9993805ca28f610df72fe36406/DEEP%20LEARNING/LSTM%20RNN/Sentiment%20analysis%20LSTM%20wth%20Bidirectional%20%20%2B%20Custom%20Attention.ipynb

# import librairies
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import Input, Dense,multiply
from keras.layers.core import *
from keras.models import *


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert len(input_shape) == 3
        assert isinstance(input_shape, list)

        self.W = self.add_weight((input_shape[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[0][-1]

        if self.bias:
            self.b = self.add_weight((input_shape[0][-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        assert isinstance(x, list)
        d, e = x
        eij_1 = d * e
        #eij = K.reshape(K.reshape(eij_1, (-1, features_dim)) * K.reshape(self.W, (features_dim, 1)), (-1, step_dim, features_dim))
        eij =eij_1 * self.W

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        #a = K.expand_dims(a)
        weighted_input = e * a
    #print weigthted_input.shape
        return weighted_input

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        #return input_shape[0],  self.features_dim
        assert isinstance(input_shape, list)
        return (input_shape[0][0],  self.step_dim, input_shape[0][-1])

