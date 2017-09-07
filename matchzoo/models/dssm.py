# -*- coding=utf-8 -*-

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
import tensorflow as tf

from model import BasicModel
import sys
sys.path.append('../matchzoo/layers/')
from SparseFullyConnectedLayer import *

class DSSM(BasicModel):
    def __init__(self, config):
        super(DSSM, self).__init__(config)
        self.__name = 'DSSM'
        self.check_list = [ 'feat_size', 
                   'num_layers', 'hidden_sizes']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print '[DSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        self.set_default('num_layers', 2)
        self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['feat_size'],))#, sparse=True)
        print('[Input] query:\t%s' % str(query.get_shape().as_list())) 
        doc = Input(name='doc', shape=(self.config['feat_size'],))#, sparse=True)
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list())) 

        def mlp_work(input_dim):
            seq = Sequential()
            seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
            for i in range(self.config['num_layers']-1):
                seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
            return seq
            
        mlp = mlp_work(self.config['feat_size'])
        rq = mlp(query)
        rd = mlp(doc)

        #out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
