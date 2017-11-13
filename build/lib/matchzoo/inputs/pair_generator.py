# -*- coding: utf-8 -*-

import sys
import random
import six
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
from keras.utils.generic_utils import deserialize_keras_object

class PairBasicGenerator(object):
    def __init__(self, config):
        self.__name = 'PairBasicGenerator'
        self.config = config
        rel_file = config['relation_train']
        self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.check_list = ['relation_train', 'batch_size']
        if config['use_iter']:
            self.pair_list_iter = self.make_pair_iter(self.rel)
            self.pair_list = []
        else:
            self.pair_list = self.make_pair_static(self.rel)
            self.pair_list_iter = None

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print '[%s] Error %s not in config' % (self.__name, e)
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
        pass

    def get_batch_iter(self):
        pass

    def get_batch(self):
        if self.config['use_iter']:
            return self.batch_iter.next()
        else:
            return self.get_batch_static()

    def get_batch_generator(self):
        pass

    @property
    def num_pairs(self):
        return len(self.pair_list)

class PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator, self).__init__(config=config)
        self.__name = 'PairGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[PairGenerator] parameter check wrong.')
        print '[PairGenerator] init done'

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
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
            
        return X1, X1_len, X2, X2_len, Y    

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32) # size of X1: batch_size * 2
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
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

class DRMM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DRMM_PairGenerator, self).__init__(config=config)
        self.__name = 'DRMM_PairGenerator'
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.embed = config['embed']
        self.hist_size = config['hist_size']
        self.fill_word = config['fill_word']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'embed', 'hist_size', 'fill_word'])
        self.use_hist_feats = False
        if 'hist_feats_file' in config:
            hist_feats = read_features(config['hist_feats_file'])
            self.hist_feats = {}
            for idx, (label, d1, d2) in enumerate(self.rel):
                self.hist_feats[(d1, d2)] = hist_feats[idx]
            self.use_hist_feats = True
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[DRMM_PairGenerator] parameter check wrong.')
        print '[DRMM_PairGenerator] init done'

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
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
            X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len
            
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
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
                    X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len
                    
                yield X1, X1_len, X2, X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

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

