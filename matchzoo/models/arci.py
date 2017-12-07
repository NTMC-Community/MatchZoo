# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

from utils.utility import *

class ARCI(BasicModel):
    def __init__(self, config):
        super(ARCI, self).__init__(config)
        self.__name = 'ARCI'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'kernel_size', 'kernel_count', 'dropout_rate',
                   'q_pool_size', 'd_pool_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCI] parameter check wrong')
        print('[ARCI] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', 3)
        self.set_default('q_pool_size', 2)
        self.set_default('d_pool_size', 2)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        q_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (q_embed)
        show_layer_info('Conv1D', q_conv1)
        d_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (d_embed)
        show_layer_info('Conv1D', d_conv1)

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        show_layer_info('MaxPooling1D', q_pool1)
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)
        show_layer_info('MaxPooling1D', d_pool1)

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])
        show_layer_info('Concatenate', pool1)

        pool1_flat = Flatten()(pool1)
        show_layer_info('Flatten', pool1_flat)

        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
