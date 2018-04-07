# -*- coding=utf-8 -*-
import sys

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from model import BasicModel

sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from layers.MatchTensor import *
from layers.SpatialGRU import *
from utils.utility import *


class MATCHSRNN(BasicModel):
    def __init__(self, config):
        super(MATCHSRNN, self).__init__(config)
        self.__name = 'MatchSRNN'
        self.check_list = ['text1_maxlen', 'text2_maxlen',
                           'embed', 'embed_size', 'vocab_size', 'channal', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.channel = config['channel']
        print self.channel
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchSRNN] parameter check wrong')
        print '[MatchSRNN] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('channal', 3)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        match_tensor = MatchTensor(channel=self.config['channel'])([q_embed, d_embed])
        show_layer_info('MatchTensor', match_tensor)
        match_tensor_permute = Permute((2, 3, 1))(match_tensor)
        h_ij = SpatialGRU()(match_tensor)
        show_layer_info('SpatialGRU', h_ij)
        h_ij_drop = Dropout(rate=self.config['dropout_rate'])(h_ij)
        show_layer_info('Dropout', h_ij_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(h_ij_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(h_ij_drop)
        show_layer_info('Dense', out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
