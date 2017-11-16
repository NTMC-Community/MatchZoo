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
sys.path.append('../matchzoo/utils/')
from utility import *
from SparseFullyConnectedLayer import *

class DSSM(BasicModel):
    def __init__(self, config):
        super(DSSM, self).__init__(config)
        self.__name = 'DSSM'
        self.check_list = [ 'feat_size', 'hidden_sizes']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print '[DSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['feat_size'],))#, sparse=True)
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['feat_size'],))#, sparse=True)
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))

        def mlp_work(input_dim):
            seq = Sequential()
            #seq.add(SparseFullyConnectedLayer(self.config['hidden_sizes'][0], input_dim=input_dim, activation='relu'))
            num_hidden_layers = len(self.config['hidden_sizes'])
            if num_hidden_layers == 1:
                seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,)))
            else:
                seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
                for i in range(num_hidden_layers-2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
                seq.add(Dense(self.config['hidden_sizes'][num_hidden_layers-1]))
            return seq

        mlp = mlp_work(self.config['feat_size'])
        rq = mlp(query)
        print('[layer]: MLP\t[shape]: %s] \n%s' % (str(rq.get_shape().as_list()), show_memory_use()))
        rd = mlp(doc)
        print('[layer]: MLP\t[shape]: %s] \n%s' % (str(rd.get_shape().as_list()), show_memory_use()))
        #out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        print('[layer]: Dot\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
