# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import keras
from keras import backend as K
from keras.layers import Lambda


def rank_hinge_loss(y_true, y_pred):
    #output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred) 
    y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
    loss = K.maximum(0., 1 + y_neg - y_pos)
    return K.mean(loss)
