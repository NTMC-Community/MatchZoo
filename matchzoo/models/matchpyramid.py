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
        config['kernel_count'] = 32
        config['kernel_size'] = [3, 3]
        config['dpool_size'] = [3, 10]
        return config

    config = default_config(config)
    check_list = [ 'text1_maxlen', 'text2_maxlen', 
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size']
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

    cross = Dot(axes=[2, 2])([q_embed, d_embed])
    cross_reshape = Reshape((config['text1_maxlen'], config['text2_maxlen'], 1))(cross)

    conv2d = Conv2D(config['kernel_count'], config['kernel_size'], padding='same', activation='relu')
    dpool = DynamicMaxPooling(config['dpool_size'][0], config['dpool_size'][1])

    conv1 = conv2d(cross_reshape)
    pool1 = dpool([conv1, dpool_index])
    pool1_flat = Flatten()(pool1)
    out_ = Dense(1)(pool1_flat)

    model = Model(inputs=[query, doc, dpool_index], outputs=out_)
    return model
