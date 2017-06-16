# -*- coding: utf-8 -*-
from __future__ import print_function

import random
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import Callback

class LossEarlyStopping(Callback)

def eval_map(y_true, y_pred, rel_thre=0):
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_pred, y_true)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[0], reverse=True)
    ipos = 0
    for j, (p,g) in enumerate(c):
        if g > rel_thre:
            ipos += 1
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s

def eval_ndcg(y_true, y_pred, k = 10):
    s = 0.
    return s

def eval_precision(y_true, y_pred, k = 10):
    s = 0.
    return s

def eval_mrr(y_true, y_pred, k = 10):
    s = 0.
    return s
