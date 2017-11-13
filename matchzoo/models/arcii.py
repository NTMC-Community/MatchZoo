# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
from DynamicMaxPooling import *
from Match import *

class ARCII(BasicModel):
    def __init__(self, config):
        super(ARCII, self).__init__(config)
        self.__name = 'ARCII'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCII] parameter check wrong')
        print '[ARCII] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', 3)
        self.set_default('dpool_size', [3, 10])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[Input] query:\t%s' % str(query.get_shape().as_list()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list()))
        dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')
        print('[Input] dpool_index:\t%s' % str(dpool_index.get_shape().as_list()))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        print('[Embedding] query_embed:\t%s' % str(q_embed.get_shape().as_list()))
        d_embed = embedding(doc)
        print('[Embedding] doc_embed:\t%s' % str(d_embed.get_shape().as_list()))

        q_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (q_embed)
        print('[Conv1D] query_conv1:\t%s' % str(q_conv1.get_shape().as_list()))
        d_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (d_embed)
        print('[Conv1D] doc_conv1:\t%s' % str(d_conv1.get_shape().as_list()))

        cross = Match(match_type='plus')([q_conv1, d_conv1])
        print('[Match: plus] cross:\t%s' % str(cross.get_shape().as_list()))

        #cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], -1))(cross)
        #print('[Reshape] cross_reshape:\t%s' % str(cross_reshape.get_shape().as_list()))

        conv2d = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')(cross)
        print('[Conv2D] conv2d:\t%s' % str(conv2d.get_shape().as_list()))
        dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])([conv2d, dpool_index])
        print('[DynamicMaxPooling] dpool:\t%s' % str(dpool.get_shape().as_list()))

        pool1_flat = Flatten()(dpool)
        print('[Flatten] pool1_flat:\t%s' % str(pool1_flat.get_shape().as_list()))
        out_ = Dense(1)(pool1_flat)
        print('[Dense] out_:\t%s' % str(out_.get_shape().as_list()))

        model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        return model
