# -*- coding=utf-8 -*-

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
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
        self.check_list = [ 'vocab_size', 'hidden_sizes', 'dropout_rate']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print '[DSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_sizes', [300, 128])
        self.set_default('dropout_rate', 0.5)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['vocab_size'],))#, sparse=True)
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['vocab_size'],))#, sparse=True)
        show_layer_info('Input', doc)

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
                    seq.add(Dropout(rate=self.config['dropout_rate']))
                seq.add(Dense(self.config['hidden_sizes'][num_hidden_layers-1]))
                seq.add(Dropout(rate=self.config['dropout_rate']))
            return seq

        mlp = mlp_work(self.config['vocab_size'])
        rq = mlp(query)
        show_layer_info('MLP', rq)
        rd = mlp(doc)
        show_layer_info('MLP', rd)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        show_layer_info('Dot', out_)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
            show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
