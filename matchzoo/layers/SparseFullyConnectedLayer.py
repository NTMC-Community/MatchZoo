# -*- conding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import InputSpec, Layer
#from keras.regularizers import ActivityRegularizer

class SparseFullyConnectedLayer(Layer):
    def __init__(self, output_dim, init='glorot_uniform', activation='relu',weights=None,
            W_regularizer=None, b_regularizer=None, activity_regularizer=None,
            W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.W_initializer = initializers.get(init)
        self.b_initializer = initializers.get('zeros')
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(SparseFullyConnectedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        #self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.input_spec = InputSpec(ndim=2, axes={1: input_dim})

        self.W = self.add_weight(
                shape=(input_dim, self.output_dim),
                initializer=self.W_initializer,
                name='SparseFullyConnected_W',
                regularizer=self.W_regularizer,
                constraint=self.W_constraint)
        self.b = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.b_initializer,
                name='SparseFullyConnected_b',
                regularizer=self.b_regularizer,
                constraint=self.b_constraint)


        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        #self.built = True
        #super(SparseFullyConnectedLayer, self).build(input_shape)

    def call(self, x, mask=None):
        #sys.stderr.write("sparse fuylly connected layer input data %s type:%s\n" % (x.name, K.type(x)))
        #sys.stderr.write("sparse fuylly connected layer weight type:%s\n" % (K.type(self.W)))
        print(str(K.ndim(x)))
        return self.activation(tf.sparse_tensor_dense_matmul(x, self.W) + self.b)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                'W_initializer':initializers.serialize(self.W_initializer),
                'b_initializer':initializers.serialize(self.W_initializer),
                'activation': activations.serialize(self.activation),
                'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                'input_dim': self.input_dim}
        base_config = super(SparseFullyConnectedLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    '''
    def create_input_layer(self, batch_input_shape,
            input_dtype=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower() + '_input_'
            name = prefix + str(K.get_uid(prefix))
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        # instantiate the input layer
        x = SparseInput(batch_shape = batch_input_shape,
                dtype = input_dtype, name = name)
        # this will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created
        self(x)
    '''
