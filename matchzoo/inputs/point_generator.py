# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import sys
import random
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
import scipy.sparse as sp

class PointGenerator(object):
    def __init__(self, config):
        self.__name = 'PointGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        rel_file = config['relation_file']
        self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.target_mode = config['target_mode']
        self.class_num = config['class_num']
        self.is_train = config['phase'] == 'TRAIN'
        self.point = 0
        self.total_rel_num = len(self.rel)
        self.check_list = ['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'relation_file', 'batch_size', 'vocab_size']
        if not self.check():
            raise TypeError('[PointGenerator] parameter check wrong.')

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True

    def get_batch(self, randomly=True):
        ID_pairs = []
        X1 = np.zeros((self.batch_size, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size,), dtype=np.int32)
        X2 = np.zeros((self.batch_size, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size,), dtype=np.int32)
        if self.target_mode == 'regression':
            Y = np.zeros((self.batch_size,), dtype=np.int32)
        elif self.target_mode == 'classification':
            Y = np.zeros((self.batch_size, self.class_num), dtype=np.int32)

        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            if randomly:
                label, d1, d2 = random.choice(self.rel)
            else:
                label, d1, d2 = self.rel[self.point]
                self.point += 1
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            X1[i, :d1_len], X1_len[i]   = self.data1[d1][:d1_len], d1_len
            X2[i, :d2_len], X2_len[i]   = self.data2[d2][:d2_len], d2_len
            if self.target_mode == 'regression':
                Y[i] = label
            elif self.target_mode == 'classification':
                Y[i, label] = 1.
            ID_pairs.append((d1, d2))
        return X1, X1_len, X2, X2_len, Y, ID_pairs

    def get_batch_generator(self):
        if self.is_train:
            while True:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch()
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)
        else:
            while self.point + self.batch_size <= self.total_rel_num:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch(randomly = False)
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID':ID_pairs}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID':ID_pairs}, Y)

    def reset(self):
        self.point = 0

class Triletter_PointGenerator(object):
    def __init__(self, config):
        self.__name = 'Triletter_PointGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.dtype = config['dtype'].lower()
        if self.dtype == 'cdssm':
            self.data1_maxlen = config['text1_maxlen']
            self.data2_maxlen = config['text2_maxlen']
        rel_file = config['relation_file']
        self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.vocab_size = config['vocab_size']
        self.fill_word = self.vocab_size - 1
        self.target_mode = config['target_mode']
        self.class_num = config['class_num']
        self.is_train = config['phase'] == 'TRAIN'
        self.point = 0
        self.total_rel_num = len(self.rel)
        self.check_list = ['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'relation_file', 'batch_size', 'vocab_size', 'dtype', 'word_triletter_map_file']
        if not self.check():
            raise TypeError('[Triletter_PointGenerator] parameter check wrong.')
        self.word_triletter_map = self.read_word_triletter_map(self.config['word_triletter_map_file'])

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True

    def read_word_triletter_map(self, wt_map_file):
        word_triletter_map = {}
        for line in open(wt_map_file):
            r = line.strip().split()
            word_triletter_map[int(r[0])] = map(int, r[1:])
        return word_triletter_map

    def map_word_to_triletter(self, words):
        triletters = []
        for wid in words:
            if wid in self.word_triletter_map:
                triletters.extend(self.word_triletter_map[wid])
        return triletters

    def transfer_feat2sparse(self, dense_feat):
        data = []
        indices = []
        indptr = [0]
        for feat in dense_feat:
            for val in feat:
                indices.append(val)
                data.append(1)
            indptr.append(indptr[-1] + len(feat))
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), self.vocab_size), dtype="float32")

    def transfer_feat2fixed(self, feats, max_len, fill_val):
        num_feat = len(feats)
        nfeat = np.zeros((num_feat, max_len), dtype=np.int32)
        nfeat[:] = fill_val
        for i in range(num_feat):
            rlen = min(max_len, len(feats[i]))
            nfeat[i,:rlen] = feats[i][:rlen]
        return nfeat

    def get_batch(self, randomly=True):
        ID_pairs = []
        X1_len = np.zeros((self.batch_size,), dtype=np.int32)
        X2_len = np.zeros((self.batch_size,), dtype=np.int32)
        if self.target_mode == 'regression':
            Y = np.zeros((self.batch_size,), dtype=np.int32)
        elif self.target_mode == 'classification':
            Y = np.zeros((self.batch_size, self.class_num), dtype=np.int32)

        X1, X2 = [], []
        for i in range(self.batch_size):
            if randomly:
                label, d1, d2 = random.choice(self.rel)
            else:
                label, d1, d2 = self.rel[self.point]
                self.point += 1
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            X1_len[i], X2_len[i]  = d1_len,  d2_len
            X1.append(self.map_word_to_triletter(self.data1[d1]))
            X2.append(self.map_word_to_triletter(self.data2[d2]))
            if self.target_mode == 'regression':
                Y[i] = label
            elif self.target_mode == 'classification':
                Y[i, label] = 1.
            ID_pairs.append((d1, d2))
        if self.dtype == 'dssm':
            return self.transfer_feat2sparse(X1).toarray(), X1_len, self.transfer_feat2sparse(X2).toarray(), X2_len, Y, ID_pairs
        elif self.dtype == 'cdssm':
            return self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word), X1_len,  \
                    self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word), X2_len, Y, ID_pairs


    def get_batch_generator(self):
        if self.is_train:
            while True:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch()
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)
        else:
            while self.point + self.batch_size <= self.total_rel_num:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch(randomly = False)
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID':ID_pairs}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID':ID_pairs}, Y)

    def reset(self):
        self.point = 0

class DRMM_PointGenerator(object):
    def __init__(self, config):
        self.__name = 'DRMM_PointGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        rel_file = config['relation_file']
        self.embed = config['embed']
        if 'bin_num' in config:
            self.hist_size = config['bin_num']
        else:
            self.hist_size = config['hist_size']
        self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.fill_word = config['vocab_size'] - 1
        self.target_mode = config['target_mode']
        self.class_num = config['class_num']
        self.is_train = config['phase'] == 'TRAIN'
        self.point = 0
        self.total_rel_num = len(self.rel)
        self.check_list = ['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'relation_file', 'batch_size', 'vocab_size']
        self.use_hist_feats = False
        if 'hist_feats_file' in config:
            hist_feats = read_features_without_id(config['hist_feats_file'])
            self.hist_feats = {}
            for idx, (label, d1, d2) in enumerate(self.rel):
                self.hist_feats[(d1, d2)] = list(hist_feats[idx])
            self.use_hist_feats = True
        if not self.check():
            raise TypeError('[DRMM_PointGenerator] parameter check wrong.')

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

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True

    def get_batch(self, randomly=True):
        ID_pairs = []
        X1 = np.zeros((self.batch_size, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size,), dtype=np.int32)
        X2 = np.zeros((self.batch_size, self.data1_maxlen, self.hist_size), dtype=np.float32)
        X2_len = np.zeros((self.batch_size,), dtype=np.int32)
        if self.target_mode == 'regression':
            Y = np.zeros((self.batch_size,), dtype=np.int32)
        elif self.target_mode == 'classification':
            Y = np.zeros((self.batch_size, self.class_num), dtype=np.int32)

        X1[:] = self.fill_word
        for i in range(self.batch_size):
            if randomly:
                label, d1, d2 = random.choice(self.rel)
            else:
                label, d1, d2 = self.rel[self.point]
                self.point += 1
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            X1[i, :d1_len], X1_len[i]   = self.data1[d1][:d1_len], d1_len
            X2[i], X2_len[i]   = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
            if self.target_mode == 'regression':
                Y[i] = label
            elif self.target_mode == 'classification':
                Y[i, label] = 1.
            ID_pairs.append((d1, d2))
        return X1, X1_len, X2, X2_len, Y, ID_pairs

    def get_batch_generator(self):
        if self.is_train:
            while True:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch()
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)
        else:
            while self.point + self.batch_size <= self.total_rel_num:
                X1, X1_len, X2, X2_len, Y, ID_pairs = self.get_batch(randomly = False)
                if self.config['use_dpool']:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID':ID_pairs}, Y)
                else:
                    yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID':ID_pairs}, Y)

    def reset(self):
        self.point = 0
