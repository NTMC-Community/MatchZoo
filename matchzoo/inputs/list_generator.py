# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import sys
import random
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
import scipy.sparse as sp

class ListBasicGenerator(object):
    def __init__(self, config={}):
        self.__name = 'ListBasicGenerator'
        self.config = config
        self.batch_list = config['batch_list']
        if 'relation_file' in config:
            self.rel = read_relation(filename=config['relation_file'])
            self.list_list = self.make_list(self.rel)
            self.num_list = len(self.list_list)
        self.check_list = []
        self.point = 0

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
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
        print('List Instance Count:', len(list_list), end='\n')
        return list(list_list.items())

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
        self.fill_word = config['vocab_size'] - 1
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen'])
        if not self.check():
            raise TypeError('[ListGenerator] parameter check wrong.')
        print('[ListGenerator] init done', end='\n')

    def get_batch(self):
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list

            bsize = sum([len(pt[1]) for pt in currbatch])
            ID_pairs = []
            list_count = [0]
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = min(self.data2_maxlen, len(d2_cont))
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j, :d2_len], X2_len[j] = d2_cont[:d2_len], d2_len
                    ID_pairs.append((d1, d2))
                    Y[j] = l
                    j += 1
            yield X1, X1_len, X2, X2_len, Y, ID_pairs, list_count

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs, list_counts in self.get_batch():
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID': ID_pairs, 'list_counts': list_counts}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs, 'list_counts': list_counts}, Y)

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls = [], [], [], [], [], []
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list

            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = min(self.data2_maxlen, len(d2_cont))
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j, :d2_len], X2_len[j] = d2_cont[:d2_len], d2_len
                    Y[j] = l
                    j += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
            list_count_ls.append(list_count)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls

class Triletter_ListGenerator(ListBasicGenerator):
    def __init__(self, config={}):
        super(Triletter_ListGenerator, self).__init__(config=config)
        self.__name = 'Triletter_ListGenerator'
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.dtype = config['dtype'].lower()
        if self.dtype == 'cdssm':
            self.data1_maxlen = config['text1_maxlen']
            self.data2_maxlen = config['text2_maxlen']
        self.vocab_size = config['vocab_size']
        self.fill_word = self.vocab_size - 1
        self.check_list.extend(['data1', 'data2', 'dtype', 'vocab_size', 'word_triletter_map_file'])
        if not self.check():
            raise TypeError('[Triletter_ListGenerator] parameter check wrong.')
        self.word_triletter_map = self.read_word_triletter_map(self.config['word_triletter_map_file'])
        print('[Triletter_ListGenerator] init done', end='\n')

    def read_word_triletter_map(self, wt_map_file):
        word_triletter_map = {}
        for line in open(wt_map_file):
            r = line.strip().split()
            word_triletter_map[int(r[0])] = list(map(int, r[1:]))
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

    def get_batch(self):
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            bsize = sum([len(pt[1]) for pt in currbatch])
            ID_pairs = []
            list_count = [0]
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1, X2 = [], []
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = len(d1_cont)
                for l, d2 in d2_list:
                    X1_len[j] = d1_len
                    X1.append(self.map_word_to_triletter(d1_cont))
                    d2_cont = list(self.data2[d2])
                    d2_len = len(d2_cont)
                    X2_len[j] = d2_len
                    X2.append(self.map_word_to_triletter(d2_cont))
                    ID_pairs.append((d1, d2))
                    Y[j] = l
                    j += 1
            if self.dtype == 'dssm':
                yield self.transfer_feat2sparse(X1).toarray(), X1_len, self.transfer_feat2sparse(X2).toarray(), X2_len, Y, ID_pairs, list_count
            elif self.dtype == 'cdssm':
                yield self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word), X1_len,  \
                    self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word), X2_len, Y, \
                    ID_pairs, list_count

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs, list_counts in self.get_batch():
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs, 'list_counts':list_counts}, Y)

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls = [], [], [], [], [], []
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1, X2 = [], []
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = len(d1_cont)
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = len(d2_cont)
                    X1_len[j] = d1_len
                    X1.append(self.map_word_to_triletter(d1_cont))
                    X2_len[j] = d2_len
                    X2.append(self.map_word_to_triletter(d2_cont))
                    Y[j] = l
                    j += 1
            if self.type == 'dssm':
                x1_ls.append(self.transfer_feat2sparse(X1).toarray())
                x2_ls.append(self.transfer_feat2sparse(X2).toarray())
            elif self.type == 'cdssm':
                x1_ls.append(self.transfer_feat2fixed(X1, self.data1_maxlen, self.fill_word))
                x2_ls.append(self.transfer_feat2fixed(X2, self.data2_maxlen, self.fill_word))
            x1_len_ls.append(X1_len)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
            list_count_ls.append(list_count)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls

class DRMM_ListGenerator(ListBasicGenerator):
    def __init__(self, config={}):
        super(DRMM_ListGenerator, self).__init__(config=config)
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.embed = config['embed']
        if 'bin_num' in config:
            self.hist_size = config['bin_num']
        else:
            self.hist_size = config['hist_size']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'embed'])
        self.use_hist_feats = False
        if 'hist_feats_file' in config:
            hist_feats = read_features_without_id(config['hist_feats_file'])
            self.hist_feats = {}
            for idx, (label, d1, d2) in enumerate(self.rel):
                self.hist_feats[(d1, d2)] = hist_feats[idx]
            self.use_hist_feats = True
        if not self.check():
            raise TypeError('[DRMM_ListGenerator] parameter check wrong.')
        print('[DRMM_ListGenerator] init done, list number: %d. ' % (self.num_list), end='\n')

    def cal_hist(self, t1, t2, data1_maxlen, hist_size):
        mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
        t1_cont = list(self.data1[t1])
        t2_cont = list(self.data2[t2])
        d1len = len(t1_cont)
        if self.use_hist_feats:
            assert (t1, t2) in self.hist_feats
            caled_hist = np.reshape(self.hist_feats[(t1, t2)], (d1len, hist_size))
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

    def get_batch(self):
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point + self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            ID_pairs = []
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data1_maxlen, self.hist_size), dtype=np.float32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                d1_len = min(self.data1_maxlen, len(d1_cont))
                list_count.append(list_count[-1] + len(d2_list))
                for l, d2 in d2_list:
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    d2_cont = list(self.data2[d2])
                    d2_len = len(d2_cont)
                    X2[j], X2_len[j] = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
                    ID_pairs.append((d1, d2))
                    Y[j] = l
                    j += 1
            yield X1, X1_len, X2, X2_len, Y, ID_pairs, list_count

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, Y, ID_pairs, list_counts in self.get_batch():
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs, 'list_counts': list_counts}, Y)
    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls = [], [], [], [], [], []
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point + self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data1_maxlen, self.hist_size), dtype=np.float32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = len(d2_cont)
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j], X2_len[j] = self.cal_hist(d1, d2, self.data1_maxlen, self.hist_size), d2_len
                    Y[j] = l
                    j += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            y_ls.append(Y)
            list_count_ls.append(list_count)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls

class ListGenerator_Feats(ListBasicGenerator):
    def __init__(self, config={}):
        super(ListGenerator_Feats, self).__init__(config=config)
        self.__name = 'ListGenerator'
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'pair_feat_size', 'pair_feat_file', 'query_feat_size', 'query_feat_file'])
        if not self.check():
            raise TypeError('[ListGenerator] parameter check wrong.')

        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.pair_feat_size = config['pair_feat_size']
        self.query_feat_size = config['query_feat_size']
        pair_feats = read_features_without_id(config['pair_feat_file'])
        self.query_feats =  read_features_with_id(config['query_feat_file'])
        self.pair_feats = {}
        for idx, (label, d1, d2) in enumerate(self.rel):
            self.pair_feats[(d1, d2)] = pair_feats[idx]

        print('[ListGenerator] init done', end='\n')

    def get_batch(self):
        while self.point < self.num_list:
            currbatch = []
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point+self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list

            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            ID_pairs = []
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            X3 = np.zeros((bsize, self.pair_feat_size), dtype=np.float32)
            X4 = np.zeros((bsize, self.query_feat_size), dtype=np.float32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1_cont = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = min(self.data2_maxlen, len(d2_cont))
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j, :d2_len], X2_len[j] = d2_cont[:d2_len], d2_len
                    X3[j, :self.pair_feat_size] = self.pair_feats[(d1, d2)]
                    X4[j, :d1_len] = self.query_feats[d1][:self.query_feat_size]
                    ID_pairs.append((d1, d2))
                    Y[j] = l
                    j += 1
            yield X1, X1_len, X2, X2_len, X3, X4, Y, ID_pairs, list_count

    def get_batch_generator(self):
        for X1, X1_len, X2, X2_len, X3, X4, Y, ID_pairs, list_counts in self.get_batch():
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'pair_feats': X3, 'query_feats': X4, 'ID': ID_pairs, 'list_counts': list_counts}, Y)

    def get_all_data(self):
        x1_ls, x1_len_ls, x2_ls, x2_len_ls, x3_ls, x4_ls, y_ls, list_count_ls = [], [], [], [], [], [], [], []
        while self.point < self.num_list:
            if self.point + self.batch_list <= self.num_list:
                currbatch = self.list_list[self.point: self.point + self.batch_list]
                self.point += self.batch_list
            else:
                currbatch = self.list_list[self.point:]
                self.point = self.num_list
            bsize = sum([len(pt[1]) for pt in currbatch])
            list_count = [0]
            X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
            X1_len = np.zeros((bsize,), dtype=np.int32)
            X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
            X2_len = np.zeros((bsize,), dtype=np.int32)
            X3 = np.zeros((bsize, self.pair_feat_size), dtype=np.float32)
            X4 = np.zeros((bsize, self.query_feat_size), dtype=np.float32)
            Y = np.zeros((bsize,), dtype= np.int32)
            X1[:] = self.fill_word
            X2[:] = self.fill_word
            j = 0
            for pt in currbatch:
                d1, d2_list = pt[0], pt[1]
                d1 = list(self.data1[d1])
                list_count.append(list_count[-1] + len(d2_list))
                d1_len = min(self.data1_maxlen, len(d1_cont))
                for l, d2 in d2_list:
                    d2_cont = list(self.data2[d2])
                    d2_len = min(self.data2_maxlen, len(d2_cont))
                    X1[j, :d1_len], X1_len[j] = d1_cont[:d1_len], d1_len
                    X2[j, :d2_len], X2_len[j] = d2_cont[:d2_len], d2_len
                    X3[j, :self.pair_feat_size] = self.pair_feats[(d1, d2)]
                    X4[j, :d1_len] = self.query_feats[d1][:self.query_feat_size]
                    Y[j] = l
                    j += 1
            x1_ls.append(X1)
            x1_len_ls.append(X1_len)
            x2_ls.append(X2)
            x2_len_ls.append(X2_len)
            x3_ls.append(X3)
            y_ls.append(Y)
            list_count_ls.append(list_count)
        return x1_ls, x1_len_ls, x2_ls, x2_len_ls, x3_ls, x4_ls,  y_ls, list_count_ls

