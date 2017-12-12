# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
import tensorflow as tf
from model import BasicModel
from utils.utility import *
from layers.SparseFullyConnectedLayer import *

class DSSM(BasicModel):
    def __init__(self, config):
        super(DSSM, self).__init__(config)
        self.__name = 'DSSM'
        self.check_list = [ 'vocab_size', 'hidden_sizes', 'dropout_rate']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print('[DSSM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_sizes', [300, 128])
        self.set_default('dropout_rate', 0.5)
        self.set_default('reg_rate', 0.0)
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
                seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,), activity_regularizer=regularizers.l2(self.config['reg_rate'])))
            else:
                seq.add(Dense(self.config['hidden_sizes'][0], activation='tanh', input_shape=(input_dim,), activity_regularizer=regularizers.l2(self.config['reg_rate'])))
                for i in range(num_hidden_layers-2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='tanh', activity_regularizer=regularizers.l2(self.config['reg_rate'])))
                    seq.add(Dropout(rate=self.config['dropout_rate']))
                seq.add(Dense(self.config['hidden_sizes'][num_hidden_layers-1], activity_regularizer=regularizers.l2(self.config['reg_rate'])))
                seq.add(Dropout(rate=self.config['dropout_rate']))
            return seq

        mlp = mlp_work(self.config['vocab_size'])
        rq = mlp(query)
        show_layer_info('MLP', rq)
        rd = mlp(doc)
        show_layer_info('MLP', rd)

        '''
        rep = Concatenate(axis=1) ([rq, rd])
        show_layer_info('Concatenate', rep)
        rep = Dropout(rate=self.config['dropout_rate'])(rep)
        show_layer_info('Dropout', rep)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(rep)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(rep)
        show_layer_info('Dense', out_)
        '''
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        show_layer_info('Dot', out_)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
            show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
