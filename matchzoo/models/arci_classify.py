# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/utils/')
from utility import *

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
        print '[ARCI] init done'

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
        print('[layer] Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[layer] Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))
        #dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        print('[layer] Embedding\t[shape]: %s] \n%s' % (str(q_embed.get_shape().as_list()), show_memory_use()))
        d_embed = embedding(doc)
        print('[layer] Embedding\t[shape]: %s] \n%s' % (str(d_embed.get_shape().as_list()), show_memory_use()))

        q_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (q_embed)
        print('[layer] Conv1D\t[shape]: %s] \n%s' % (str(q_conv1.get_shape().as_list()), show_memory_use()))
        d_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (d_embed)
        print('[layer] Conv1D\t[shape]: %s] \n%s' % (str(d_conv1.get_shape().as_list()), show_memory_use()))

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        print('[layer] MaxPooling1D\t[shape]: %s] \n%s' % (str(q_pool1.get_shape().as_list()), show_memory_use()))
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)
        print('[layer] MaxPooling1D\t[shape]: %s] \n%s' % (str(d_pool1.get_shape().as_list()), show_memory_use()))

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])
        print('[layer] Concatenate\t[shape]: %s] \n%s' % (str(pool1.get_shape().as_list()), show_memory_use()))

        pool1_flat = Flatten()(pool1)
        print('[layer] Flatten\t[shape]: %s] \n%s' % (str(pool1_flat.get_shape().as_list()), show_memory_use()))
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        print('[layer] Dropout\t[shape]: %s] \n%s' % (str(pool1_flat_drop.get_shape().as_list()), show_memory_use()))
        out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        print('[layer] Dense\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
