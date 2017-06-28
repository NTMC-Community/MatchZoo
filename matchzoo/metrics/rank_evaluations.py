# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import random
import numpy as np
import math
import keras
from keras import backend as K
from keras.callbacks import Callback
sys.path.append('./utils/')
from rank_data_generator import *

class LossEarlyStopping(Callback):
    def __init__(self, metric, value, mode='less'):
        super().__init__()
        self.metric = metric
        self.value = value
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        if self.mode == 'less':
            if logs[self.metric] < self.value:
                self.model.stop_training = True
                print('Early stopping -{} is {} than {}'.format(self.metric, self.mode, self.value))
            if self.mode == 'more':
                if logs[self.metric] > self.value:
                    self.model.stop_training = True
                    print('Early stopping - {} is {} than {}'.format(self.metric, self.mode, self.value))

class rank_eval():
    def __init__(self, rel_threshold=0.):
        self.rel_threshold = rel_threshold
    def zipped(self, y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_pred, y_true)
        random.shuffle(c)
        return c

    def eval(self, y_true, y_pred, 
            metrics=['map', 'p@1', 'p@5', 'p@10', 'p@20', 
                'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20'], k = 20):
        res = {}
        res['map'] = self.map(y_true, y_pred)
        all_ndcg = self.ndcg(y_true, y_pred, k=k)
        all_precision = self.precision(y_true, y_pred, k=k)
        res.update({'p@%d'%(i+1):all_precision[i] for i in range(k)})
        res.update({'ndcg@%d'%(i+1):all_ndcg[i] for i in range(k)})
        ret = {k:v for k,v in res.items() if k in metrics}
        return ret
    def map(self, y_true, y_pred):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        s = 0.
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                ipos += 1
                s += ipos / ( i + 1)
        if ipos == 0:
            return 0.
        else:
            return s / ipos
    def ndcg(self, y_true, y_pred, k = 20):
        s = 0.
        c = self.zipped(y_true, y_pred)
        c_g = sorted(c, key=lambda x:x[0], reverse=True)
        c_p = sorted(c, key=lambda x:x[1], reverse=True)
        #idcg = [0. for i in range(k)]
        idcg = np.zeros([k], dtype=np.float32)
        dcg = np.zeros([k], dtype=np.float32)
        #dcg = [0. for i in range(k)]
        for i, (g,p) in enumerate(c_g):
            if g > self.rel_threshold:
                idcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for i, (g,p) in enumerate(c_p):
            if g > self.rel_threshold:
                dcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for idx, v in enumerate(idcg):
            if v == 0.:
                dcg[idx] = 0.
            else:
                dcg[idx] /= v
        return dcg
    def precision(self, y_true, y_pred, k = 20):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        s = 0.
        precision = [0. for i in range(k)]
        precision = np.zeros([k], dtype=np.float32) #[0. for i in range(k)]
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                precision[i:] += 1
        precision = [v / (idx + 1) for idx, v in enumerate(precision)]
        return precision

'''
class MAP_eval(Callback):
    def __init__(self, validation_data, rel_threshold = 0):
        self.validation_data = validation_data
        self.maps = []
        self.rel_threshold = rel_threshold
    def eval_map(self):
        map = 0.
        inum = 0
        print(self.validation_data)
        for (x1, x1_len, x2, x2_len, y_true) in self.validation_data.get_batch:
            y_pred = self.model.predict({'query':x1, 'doc':x2})
            y_pred = list(np.squeeze(y_pred))
            zipped = zip(y_true, y_pred)
            print(zipped)
            zipped.sort(key=lambda x:x[1], reverse=True)
            curr_map = 0.
            ipos = 0
            for i, (g, p) in enumerate(*zipped):
                if g > self.rel_threshold:
                    ipos += 1
                    curr_map += 1. * ipos / ( 1. + j)
            if ipos == 0:
                score += 0.
            else:
                map += curr_map / ipos
            inum += 1
        if inum == 0:
            return 0.
        else:
            return map / inum
    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print('MAP for epoch %d is %f'%(epoch, score))
        self.maps.append(score)
'''
class MAP_eval(Callback):
    def __init__(self, x1_ls, x2_ls, y_ls, rel_threshold = 0):
        self.x1_ls = x1_ls
        self.x2_ls = x2_ls
        self.y_ls = y_ls
        self.num_list = len(x1_ls)
        self.maps = []
        self.rel_threshold = rel_threshold
    def eval_map(self):
        map = 0.
        inum = 0
        for i in range(self.num_list):
            x1 = self.x1_ls[i]
            x2 = self.x2_ls[i]
            y_true = self.y_ls[i]
            y_pred = self.model.predict({'query':x1, 'doc':x2})
            y_pred = list(np.squeeze(y_pred))
            zipped = zip(y_true, y_pred)
            zipped.sort(key=lambda x:x[1], reverse=True)
            curr_map = 0.
            ipos = 0
            for j, (g, p) in enumerate(zipped):
                if g > self.rel_threshold:
                    ipos += 1
                    curr_map += 1. * ipos / ( 1. + j)
            if ipos == 0:
                score += 0.
            else:
                map += curr_map / ipos
            inum += 1
        if inum == 0:
            return 0.
        else:
            return map / inum
    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print('MAP for epoch %d is %f'%(epoch, score))
        self.maps.append(score)

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
