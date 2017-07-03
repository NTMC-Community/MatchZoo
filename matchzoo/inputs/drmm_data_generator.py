# -*- coding: utf-8 -*-

import os
import sys
import six
import random
import numpy as np
#sys.path.append('../utils/')
#from rank_io import *
from utils.rank_io import *

        
def cal_hist(t1_rep, t2_rep, data1_maxlen, hist_size):
    mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))
    for (i,j), v in np.ndenumerate(mm):
        if i >= data1_maxlen:
            break
        vid = int((v + 1.) / 2. * ( hist_size - 1.))
        mhist[i][vid] += 1.
    mhist += 1.
    mhist = np.log10(mhist)
    return mhist

class DRMM_PairGenerator():
    def __init__(self, data1, data2, config):
        self.config = config
        rel_file = config['relation_train']
        rel = read_relation(filename=rel_file)
        self.data1 = data1
        self.data2 = data2
        self.embed = config['embed']
        self.batch_size = config['batch_size']
        self.hist_size = config['hist_size']
        self.data1_maxlen = config['text1_maxlen']
        self.fill_word = config['fill_word']
        if config['use_iter']:
            self.pair_list_iter = self.make_pair_iter(rel)
            self.pair_list = []
            self.batch_iter = self.get_batch_iter()
        else:
            self.pair_list = self.make_pair_static(rel)
            self.pair_list_iter = None

    def make_pair_static(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse = True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx+1:]:
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append( (d1, high_d2, low_d2) )
        print 'Pair Instance Count:', len(pair_list)
        return pair_list

    def make_pair_iter(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        
        while True:
            rel_set_sample = random.sample(rel_set.keys(), self.config['query_per_iter'])

            for d1 in rel_set_sample:
                label_list = sorted(rel_set[d1].keys(), reverse = True)
                for hidx, high_label in enumerate(label_list[:-1]):
                    for low_label in label_list[hidx+1:]:
                        for high_d2 in rel_set[d1][high_label]:
                            for low_d2 in rel_set[d1][low_label]:
                                pair_list.append( (d1, high_d2, low_d2) )
            #print 'Pair Instance Count:', len(pair_list)
            yield pair_list


    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data1_maxlen, self.hist_size), dtype=np.float32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        #X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = len(self.data2[d2p])
            d2n_len = len(self.data2[d2n])
            d1_embed = self.embed[self.data1[d1]]
            d2p_embed = self.embed[self.data2[d2p]]
            d2n_embed = self.embed[self.data2[d2n]]
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2], X2_len[i*2]   = cal_hist(d1_embed, d2p_embed, self.data1_maxlen, self.hist_size), d2p_len
            X2[i*2+1], X2_len[i*2+1] = cal_hist(d1_embed, d2n_embed, self.data1_maxlen, self.hist_size), d2n_len
            
        return X1, X1_len, X2, X2_len, Y    

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data1_maxlen, self.hist_size), dtype=np.float32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                #X2[:] = 0.
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = len(self.data2[d2p])
                    d2n_len = len(self.data2[d2n])
                    d1_embed = self.embed[self.data1[d1]]
                    d2p_embed = self.embed[self.data2[d2p]]
                    d2n_embed = self.embed[self.data2[d2n]]
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2], X2_len[i*2]   = cal_hist(d1_embed, d2p_embed, self.data1_maxlen, self.hist_size), d2p_len
                    X2[i*2+1], X2_len[i*2+1] = cal_hist(d1_embed, d2n_embed, self.data1_maxlen, self.hist_size), d2n_len
                    
                yield X1, X1_len, X2, X2_len, Y

    def get_batch(self):
        if self.config['use_iter']:
            return self.batch_iter.next()
        else:
            return self.get_batch_static()

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

    @property
    def num_pairs(self):
        return len(self.pair_list)

class DRMM_ListGenerator():
    def __init__(self, data1=None, data2=None, config={}):
        self.config = config
        if 'relation_test' in config:
            rel = read_relation(filename=config['relation_test'])
            self.list_list = self.make_list(rel)
            self.num_list = len(self.list_list)
        self.data1 = data1
        self.data2 = data2
        self.embed = config['embed']
        self.hist_size = config['hist_size']
        self.data1_maxlen = config['text1_maxlen']
        self.fill_word = config['fill_word']
        self.point = 0

    def make_list(self, rel):
        list_list = {}
        for label, d1, d2 in rel:
            if d1 not in list_list:
                list_list[d1] = []
            list_list[d1].append( (label, d2) )
        for d1 in list_list:
            list_list[d1] = sorted(list_list[d1], reverse = True)
        print 'List Instance Count:', len(list_list)
        return list_list.items()

    def get_batch(self):
        for point in range(self.num_list):
            ID_pairs = []
            d1, d2_list = self.list_list[point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data1_maxlen, self.hist_size), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d1_embed = self.embed[self.data1[d1]]
            for j, (l, d2) in enumerate(d2_list):
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                d2_len = len(self.data2[d2])
                d2_embed = self.embed[self.data2[d2]]
                X2[j], X2_len[j] = cal_hist(d1_embed, d2_embed, self.data1_maxlen, self.hist_size), d2_len
                ID_pairs.append((d1, d2))
                Y[j] = l
            yield X1, X1_len, X2, X2_len, Y, ID_pairs

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs in self.get_batch():
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs}, Y)

    def reset(self):
        self.point = 0

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls = [], [], [], [], []
        while self.point < self.num_list:
            d1, d2_list = self.list_list[self.point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data1_maxlen, self.hist_size), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d1_embed = self.embed[self.data1[d1]]
            for j, (l, d2) in enumerate(d2_list):
                d2_len = len(self.data2[d2])
                d2_embed = self.embed[self.data2[d2]]
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j], X2_len[j] = cal_hist(d1_embed, d2_embed, self.data1_maxlen, self.hist_size), d2_len
                Y[j] = l
            self.point += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls

