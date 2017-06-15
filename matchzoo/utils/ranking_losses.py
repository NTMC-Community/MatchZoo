# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import keras
from keras import backend as K


def rank_hinge_loss(y_pos, y_neg):
    loss = K.maximum(0., 1 + y_neg - y_pos)
    return K.mean(loss)
