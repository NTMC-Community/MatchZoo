# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import random
import numpy as np
import math

class rank_eval():

    def __init__(self, rel_threshold=0.):
        self.rel_threshold = rel_threshold

    def zipped(self, y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
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
        ipos = 0.
        s = 0.
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                ipos += 1.
                s += ipos / ( 1. + i )
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
        precision = np.zeros([k], dtype=np.float32) #[0. for i in range(k)]
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                precision[i:] += 1
            if i >= k:
                break
        precision = [v / (idx + 1) for idx, v in enumerate(precision)]
        return precision


def eval_map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s

def eval_ndcg(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x:x[0], reverse=True)
    c_p = sorted(c, key=lambda x:x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g,p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
    for i, (g,p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

def eval_precision(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    precision = 0.
    for i, (g,p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /=  k
    return precision

def eval_mrr(y_true, y_pred, k = 10):
    s = 0.
    return s
