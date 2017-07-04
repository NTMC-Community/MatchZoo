# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

import sys
sys.path.append('../matchzoo/layers/')
from DynamicMaxPooling import *

def check(config):

    def default_config(config):
        config['kernel_count'] = 64
        config['kernel_size'] = 3
        config['q_pool_size'] = 2
        config['d_pool_size'] = 2
        return config

    config = default_config(config)
    check_list = [ 'text1_maxlen', 'text2_maxlen', 
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'q_pool_size', 'd_pool_size']
    for e in check_list:
        if e not in config:
            print '[Model] Error %s not in config' % e
            return False
    return True

def build(config):
    query = Input(name='query', shape=(config['text1_maxlen'],))
    doc = Input(name='doc', shape=(config['text2_maxlen'],))
    dpool_index = Input(name='dpool_index', shape=[config['text1_maxlen'], config['text2_maxlen'], 3], dtype='int32')

    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[config['embed']], trainable = False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    q_conv1 = Conv1D(config['kernel_count'], config['kernel_size'], padding='same') (q_embed)
    d_conv1 = Conv1D(config['kernel_count'], config['kernel_size'], padding='same') (d_embed)

    q_pool1 = MaxPooling1D(pool_size=config['q_pool_size']) (q_conv1)
    d_pool1 = MaxPooling1D(pool_size=config['d_pool_size']) (d_conv1)

    pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])

    pool1_flat = Flatten()(pool1)
    out_ = Dense(1)(pool1_flat)

    model = Model(inputs=[query, doc, dpool_index], outputs=out_)
    return model
