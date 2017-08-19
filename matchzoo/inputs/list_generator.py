# -*- coding: utf-8 -*-

import sys
import random
import six
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
from keras.utils.generic_utils import deserialize_keras_object

class ListBasicGenerator(object):
    def __init__(self, config={}):
        self.__name = 'ListBasicGenerator'
        self.config = config
        if 'relation_test' in config:
            self.rel = read_relation(filename=config['relation_test'])
            self.list_list = self.make_list(self.rel)
            self.num_list = len(self.list_list)
        self.check_list = []
        self.point = 0

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print '[%s] Error %s not in config' % (self.__name, e)
                return False
        return True

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
        pass

    def get_batch_generator(self):
        pass

    def reset(self):
        self.point = 0

    def get_all_data(self):
        pass
class ListGenerator(ListBasicGenerator):
    def __init__(self, config={}):
        super(ListGenerator, self).__init__(config=config)
        self.__name = 'ListGenerator'
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen'])
        if not self.check():
            raise TypeError('[ListGenerator] parameter check wrong.')
        print '[ListGenerator] init done'

    def get_batch(self):
        for point in range(self.num_list):
            ID_pairs = []
            d1, d2_list = self.list_list[point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            for j, (l, d2) in enumerate(d2_list):
                d2_len = min(self.data2_maxlen, len(self.data2[d2]))
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j, :d2_len], X2_len[j] = self.data2[d2][:d2_len], d2_len
                ID_pairs.append((d1, d2))
                Y[j] = l
            yield X1, X1_len, X2, X2_len, Y, ID_pairs

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs in self.get_batch():
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID': ID_pairs}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs}, Y)

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls = [], [], [], [], []
        while self.point < self.num_list:
            d1, d2_list = self.list_list[self.point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            for j, (l, d2) in enumerate(d2_list):
                d2_len = min(self.data2_maxlen, len(self.data2[d2]))
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j, :d2_len], X2_len[j] = self.data2[d2][:d2_len], d2_len
                Y[j] = l
            self.point += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls

class DRMM_ListGenerator(ListBasicGenerator):
    def __init__(self, config={}):
        super(DRMM_ListGenerator, self).__init__(config=config)
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.embed = config['embed']
        self.hist_size = config['hist_size']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word', 'embed', 'hist_size'])
        self.use_hist_feats = False
        if 'hist_feats_file' in config:
            hist_feats = read_features(config['hist_feats_file'])
            self.hist_feats = {}
            for idx, (label, d1, d2) in enumerate(self.rel):
                self.hist_feats[(d1, d2)] = hist_feats[idx]
            self.use_hist_feats = True
        if not self.check():
            raise TypeError('[DRMM_ListGenerator] parameter check wrong.')
        print '[DRMM_ListGenerator] init done, list number: %d. ' % (self.num_list)

    def cal_hist(self, t1, t2, data1_maxlen, hist_size):
        mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
        d1len = len(self.data1[t1]) 
        if self.use_hist_feats:
            assert (t1, t2) in self.hist_feats
            caled_hist = np.reshape(self.hist_feats[(t1, t2)], (d1len, hist_size))
            if d1len < data1_maxlen:
                mhist[:d1len, :] = caled_hist[:, :]
            else:
                mhist[:, :] = caled_hist[:data1_maxlen, :]
        else:
            t1_rep = self.embed[self.data1[t1]]
            t2_rep = self.embed[self.data2[t2]]
            mm = t1_rep.dot(np.transpose(t2_rep))
            for (i,j), v in np.ndenumerate(mm):
                if i >= data1_maxlen:
                    break
                vid = int((v + 1.) / 2. * ( hist_size - 1.))
                mhist[i][vid] += 1.
            mhist += 1.
            mhist = np.log10(mhist)
        return mhist

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
            for j, (l, d2) in enumerate(d2_list):
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                d2_len = len(self.data2[d2])
                X2[j], X2_len[j] = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
                ID_pairs.append((d1, d2))
                Y[j] = l
            yield X1, X1_len, X2, X2_len, Y, ID_pairs

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs in self.get_batch():
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs}, Y)
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
            for j, (l, d2) in enumerate(d2_list):
                d2_len = len(self.data2[d2])
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j], X2_len[j] = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
                Y[j] = l
            self.point += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls

class ListGenerator_Feats(ListBasicGenerator):
    def __init__(self, config={}):
        super(ListGenerator_Feats, self).__init__(config=config)
        self.__name = 'ListGenerator'
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'pair_feat_size', 'pair_feat_file'])
        if not self.check():
            raise TypeError('[ListGenerator] parameter check wrong.')

        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.pair_feat_size = config['pair_feat_size']
        pair_feats = read_features(config['pair_feat_file'])
        self.pair_feats = {}
        for idx, (label, d1, d2) in enumerate(self.rel):
            self.pair_feats[(d1, d2)] = pair_feats[idx]

        print '[ListGenerator] init done'

    def get_batch(self):
        for point in range(self.num_list):
            ID_pairs = []
            d1, d2_list = self.list_list[point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            X3 = np.zeros((len(d2_list), self.pair_feat_size), dtype=np.float32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            for j, (l, d2) in enumerate(d2_list):
                d2_len = min(self.data2_maxlen, len(self.data2[d2]))
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j, :d2_len], X2_len[j] = self.data2[d2][:d2_len], d2_len
                X3[j, :self.pair_feat_size] = self.pair_feats[(d1, d2)]
                ID_pairs.append((d1, d2))
                Y[j] = l
            yield X1, X1_len, X2, X2_len, X3, Y, ID_pairs

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, X3, Y, ID_pairs in self.get_batch():
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'pair_feats': X3, 'ID': ID_pairs}, Y)

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, x3_ls, y_ls = [], [], [], [], [], []
        while self.point < self.num_list:
            d1, d2_list = self.list_list[self.point]
            X1 = np.zeros((len(d2_list), self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            X3 = np.zeros((len(d2_list), self.pair_feat_size), dtype=np.float32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            for j, (l, d2) in enumerate(d2_list):
                d2_len = min(self.data2_maxlen, len(self.data2[d2]))
                X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
                X2[j, :d2_len], X2_len[j] = self.data2[d2][:d2_len], d2_len
                X3[j, :self.pair_feat_size] = self.pair_feats[(d1, d2)]
                Y[j] = l
            self.point += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            x3_ls.append(X3)
            y_ls.append(Y)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, x3_ls,  y_ls

def serialize(generator):
    return generator.__name__

def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')

def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
