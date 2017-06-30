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

    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[config['embed']], trainable = False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    z = Dot(axes=[2, 2])([q_embed, d_embed])
    z = Reshape((config['text1_maxlen'], config['text2_maxlen'], 1))(z)
    conv2d = Conv2D(config['kernel_count'], config['kernel_size'], padding='same', activation='relu')
    dpool = DynamicMaxPooling(config['dpool_size'][0], config['dpool_size'][1])
    z = conv2d(z)
    z = dpool(z)
    z = Flatten()(z)
    out_ = Dense(1)(z)

    model = Model(inputs=[query, doc], outputs=out_)
    return model
