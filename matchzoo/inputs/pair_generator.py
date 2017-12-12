# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import sys
import random
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
import scipy.sparse as sp

class PairBasicGenerator(object):
    def __init__(self, config):
        self.__name = 'PairBasicGenerator'
        self.config = config
        rel_file = config['relation_file']
        self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.check_list = ['relation_file', 'batch_size']
        self.point = 0
        if config['use_iter']:
            self.pair_list_iter = self.make_pair_iter(self.rel)
            self.pair_list = []
        else:
            self.pair_list = self.make_pair_static(self.rel)
            self.pair_list_iter = None

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True
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
        print('Pair Instance Count:', len(pair_list), end='\n')
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
            yield pair_list

    def get_batch_static(self):
        pass

    def get_batch_iter(self):
        pass

    def get_batch(self):
        if self.config['use_iter']:
            return next(self.batch_iter)
        else:
            return self.get_batch_static()

    def get_batch_generator(self):
        pass

    @property
    def num_pairs(self):
        return len(self.pair_list)

    def reset(self):
        self.point = 0

class PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator, self).__init__(config=config)
        self.__name = 'PairGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[PairGenerator] parameter check wrong.')
        print('[PairGenerator] init done', end='\n')

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_cont = list(self.data1[d1])
            d2p_cont = list(self.data2[d2p])
            d2n_cont = list(self.data2[d2n])
            d1_len = min(self.data1_maxlen, len(d1_cont))
            d2p_len = min(self.data2_maxlen, len(d2p_cont))
            d2n_len = min(self.data2_maxlen, len(d2n_cont))
            X1[i*2,   :d1_len],  X1_len[i*2]   = d1_cont[:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = d2p_cont[:d2p_len], d2p_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = d1_cont[:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = d2n_cont[:d2n_len], d2n_len

        return X1, X1_len, X2, X2_len, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = next(self.pair_list_iter)
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                X2[:] = self.fill_word
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(list(self.data1[d1])))
                    d2p_len = min(self.data2_maxlen, len(list(self.data2[d2p])))
                    d2n_len = min(self.data2_maxlen, len(list(self.data2[d2n])))
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len

                yield X1, X1_len, X2, X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class Triletter_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(Triletter_PairGenerator, self).__init__(config=config)
        self.__name = 'Triletter_PairGenerator'
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.dtype = config['dtype'].lower()
        if self.dtype == 'cdssm':
            self.data1_maxlen = config['text1_maxlen']
            self.data2_maxlen = config['text2_maxlen']
        self.vocab_size = config['vocab_size']
        self.fill_word = self.vocab_size - 1
        self.check_list.extend(['data1', 'data2', 'dtype', 'vocab_size', 'word_triletter_map_file'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[Triletter_PairGenerator] parameter check wrong.')
        self.word_triletter_map = self.read_word_triletter_map(self.config['word_triletter_map_file'])
        print('[Triletter_PairGenerator] init done', end='\n')

    def read_word_triletter_map(self, wt_map_file):
        word_triletter_map = {}
        for line in open(wt_map_file):
            r = line.strip().split()
            word_triletter_map[int(r[0])] = list(map(int, r[1:]))
        return word_triletter_map

    def map_word_to_triletter(self, words):
        triletters = []
        for wid in words:
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
        res = sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), self.vocab_size), dtype="float32")
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), self.vocab_size), dtype="float32")

    def transfer_feat2fixed(self, feats, max_len, fill_val):
        num_feat = len(feats)
        nfeat = np.zeros((num_feat, max_len), dtype=np.int32)
        nfeat[:] = fill_val
        for i in range(num_feat):
            rlen = min(max_len, len(feats[i]))
            nfeat[i,:rlen] = feats[i][:rlen]
        return nfeat

    def get_batch_static(self):
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1, X2 = [], []
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = len(list(self.data1[d1]))
            d2p_len = len(list(self.data2[d2p]))
            d2n_len = len(list(self.data2[d2n]))
            X1_len[i*2], X1_len[i*2+1]  = d1_len,  d1_len
            X2_len[i*2], X2_len[i*2+1]  = d2p_len, d2n_len
            X1.append(self.map_word_to_triletter(self.data1[d1]))
            X1.append(self.map_word_to_triletter(self.data1[d1]))
            X2.append(self.map_word_to_triletter(self.data2[d2p]))
            X2.append(self.map_word_to_triletter(self.data2[d2n]))
        if self.dtype == 'dssm':
            return self.transfer_feat2sparse(X1).toarray(), X1_len, self.transfer_feat2sparse(X2).toarray(), X2_len, Y
        elif self.dtype == 'cdssm':
            return self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word), X1_len,  \
                    self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word), X2_len, Y


    def get_batch_iter(self):
        while True:
            self.pair_list = next(self.pair_list_iter)
            for _ in range(self.config['batch_per_iter']):
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1, X2 = [], []
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_cont = list(self.data1[d1])
                    d2p_cont = list(self.data2[d2p])
                    d2n_cont = list(self.data2[d2n])
                    d1_len = len(d1_cont)
                    d2p_len = len(d2p_cont)
                    d2n_len = len(d2n_cont)
                    X1_len[i*2],  X1_len[i*2+1]   = d1_len, d1_len
                    X2_len[i*2],  X2_len[i*2+1]   = d2p_len, d2n_len
                    X1.append(self.map_word_to_triletter(d1_cont))
                    X1.append(self.map_word_to_triletter(d1_cont))
                    X2.append(self.map_word_to_triletter(d2p_cont))
                    X2.append(self.map_word_to_triletter(d2n_cont))
                if self.dtype == 'dssm':
                    yield self.transfer_feat2sparse(X1).toarray(), X1_len, self.transfer_feat2sparse(X2).toarray(), X2_len, Y
                elif self.dtype == 'cdssm':
                    yield self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word), X1_len, \
                            self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word), X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class DRMM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DRMM_PairGenerator, self).__init__(config=config)
        self.__name = 'DRMM_PairGenerator'
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.embed = config['embed']
        if 'bin_num' in config:
            self.hist_size = config['bin_num']
        else:
            self.hist_size = config['hist_size']
        self.fill_word = config['vocab_size'] - 1
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'embed'])
        self.use_hist_feats = False
        if 'hist_feats_file' in config:
            hist_feats = read_features_without_id(config['hist_feats_file'])
            self.hist_feats = {}
            for idx, (label, d1, d2) in enumerate(self.rel):
                self.hist_feats[(d1, d2)] = hist_feats[idx]
            self.use_hist_feats = True
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[DRMM_PairGenerator] parameter check wrong.')
        print('[DRMM_PairGenerator] init done', end='\n')

    def cal_hist(self, t1, t2, data1_maxlen, hist_size):
        mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
        t1_cont = list(self.data1[t1])
        t2_cont = list(self.data2[t2])
        d1len = len(t1_cont)
        if self.use_hist_feats:
            assert (t1, t2) in self.hist_feats
            curr_pair_feats = list(self.hist_feats[(t1, t2)])
            caled_hist = np.reshape(curr_pair_feats, (d1len, hist_size))
            if d1len < data1_maxlen:
                mhist[:d1len, :] = caled_hist[:, :]
            else:
                mhist[:, :] = caled_hist[:data1_maxlen, :]
        else:
            t1_rep = self.embed[t1_cont]
            t2_rep = self.embed[t2_cont]
            mm = t1_rep.dot(np.transpose(t2_rep))
            for (i,j), v in np.ndenumerate(mm):
                if i >= data1_maxlen:
                    break
                vid = int((v + 1.) / 2. * ( hist_size - 1.))
                mhist[i][vid] += 1.
            mhist += 1.
            mhist = np.log10(mhist)
        return mhist

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data1_maxlen, self.hist_size), dtype=np.float32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_cont = list(self.data1[d1])
            d2p_cont = list(self.data2[d2p])
            d2n_cont = list(self.data2[d2n])
            d1_len = min(self.data1_maxlen, len(d1_cont))
            d2p_len = len(d2p_cont)
            d2n_len = len(d2n_cont)
            X1[i*2,   :d1_len],  X1_len[i*2]   = d1_cont[:d1_len],   d1_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = d1_cont[:d1_len],   d1_len
            X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
            X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len

        return X1, X1_len, X2, X2_len, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = next(self.pair_list_iter)
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
                    d1_cont = list(self.data1[d1])
                    d2p_cont = list(self.data2[d2p])
                    d2n_cont = list(self.data2[d2n])
                    d1_len = min(self.data1_maxlen, len(d1_cont))
                    d2p_len = len(d2p_cont)
                    d2n_len = len(d2n_cont)
                    X1[i*2,   :d1_len],  X1_len[i*2]   = d1_cont[:d1_len],   d1_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = d1_cont[:d1_len],   d1_len
                    X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
                    X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len

                yield X1, X1_len, X2, X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class PairGenerator_Feats(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator_Feats, self).__init__(config=config)
        self.__name = 'PairGenerator'
        self.config = config
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'pair_feat_size', 'pair_feat_file', 'query_feat_size', 'query_feat_file'])
        if not self.check():
            raise TypeError('[PairGenerator] parameter check wrong.')

        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.pair_feat_size = config['pair_feat_size']
        self.query_feat_size = config['query_feat_size']
        pair_feats = read_features_without_id(config['pair_feat_file'])
        self.query_feats = read_features_with_id(config['query_feat_file'])
        self.pair_feats = {}
        for idx, (label, d1, d2) in enumerate(self.rel):
            self.pair_feats[(d1, d2)] = pair_feats[idx]
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        print('[PairGenerator] init done', end='\n')

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X3 = np.zeros((self.batch_size * 2, self.pair_feat_size), dtype=np.float32)
        X4 = np.zeros((self.batch_size * 2, self.query_feat_size), dtype=np.float32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
            X3[i*2,   :self.pair_feat_size]    = self.pair_feats[(d1, d2p)][:self.pair_feat_size]
            X4[i*2,   :self.query_feat_size] = self.query_feats[d1][:self.query_feat_size]
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
            X3[i*2+1, :self.pair_feat_size]    = self.pair_feats[(d1, d2n)][:self.pair_feat_size]
            X4[i*2+1, :self.query_feat_size] = self.query_feats[d1][:self.query_feat_size]

        return X1, X1_len, X2, X2_len, X3, X4, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = next(self.pair_list_iter)
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X3 = np.zeros((self.batch_size*2, self.pair_feat_size), dtype=np.float32)
                X4 = np.zeros((self.batch_size*2, self.query_feat_size), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                X2[:] = self.fill_word
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
                    d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
                    X3[i*2,   :self.pair_feat_size]    = self.pair_feats[(d1, d2p)][:self.pair_feat_size]
                    X4[i*2,   :d1_len] = self.query_feats[d1][:self.query_feat_size]
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
                    X3[i*2+1, :self.pair_feat_size]    = self.pair_feats[(d1, d2n)][:self.pair_feat_size]
                    X4[i*2+1, :d1_len] = self.query_feats[d1][:self.query_feat_size]

                yield X1, X1_len, X2, X2_len, X3, X4, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, X3, X4, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'query_feats': X4, 'pair_feats': X3}, Y)

