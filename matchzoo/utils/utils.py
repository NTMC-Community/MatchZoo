
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import six
import keras
from keras import backend as K
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object
from keras.callbacks import Callback

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.output = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.output.append(logs.get('dense_1'))
