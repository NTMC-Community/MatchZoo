# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import six
import keras
from keras import backend as K
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object
from keras import optimizers

if K.backend() == 'tensorflow':
    import tensorflow as tf

def serialize(optimizer):
    return serialize_keras_object(optimizer)

def deserialize(config, custom_objects=None):
    """Inverse of the `serialize` function.
    # Arguments
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping
            names (strings) to custom objects
            (classes and functions)
            to be considered during deserialization.
    # Returns
        A Keras Optimizer instance.
    """
    all_classes = {
        'sgd': optimizers.SGD,
        'rmsprop': optimizers.RMSprop,
        'adagrad': optimizers.Adagrad,
        'adadelta': optimizers.Adadelta,
        'adam': optimizers.Adam,
        'adamax': optimizers.Adamax,
        'nadam': optimizers.Nadam,
        'tfoptimizer': optimizers.TFOptimizer
    }
    # Make deserialization case-insensitive for built-in optimizers.
    if config['class_name'].lower() in all_classes:
        config['class_name'] = config['class_name'].lower()
    return deserialize_keras_object(config,
                                    module_objects=all_classes,
                                    custom_objects=custom_objects,
                                    printable_module_name='optimizer')

def get(identifier):
    """Retrieves a Keras Optimizer instance.
    # Arguments
        identifier: Optimizer identifier, one of
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).
            - TensorFlow Optimizer instance
                (it will be wrapped as a Keras Optimizer).
    # Returns
        A Keras Optimizer instance.
    # Raises
        ValueError: If `identifier` cannot be interpreted.
    """
    if K.backend() == 'tensorflow':
        # Wrap TF optimizer instances
        if isinstance(identifier, tf.train.Optimizer):
            return TFOptimizer(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    if isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError('Could not interpret optimizer identifier:',
identifier)
