# -*- coding: utf-8 -*=
import numpy as np

import os
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam


MAX_Q_LEN = 5
MAX_D_LEN = 50

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding

def matching(lr=0.1):
    query = Input(name='query', shape=(2,))
    print K.int_shape(query)
    doc = Input(name='doc', shape=(3,))
    print K.int_shape(doc)
    embedding = Embedding(2000, 1)
    q_embed = embedding(query)
    d_embed = embedding(doc)
    print K.int_shape(q_embed)
    print K.int_shape(d_embed)

    #dot = K.batch_dot(q_embed, d_embed, axes=[2, 2])
    #dot = Merge([q_embed, d_embed], mode='dot', dot_axes=1)
    dot = Dot(axes=[2, 2])([q_embed, d_embed])
    out_ = Dropout(0.2)(dot)
    print K.int_shape(out_)

    #lam = Lambda( lambda x: x, input_shape=(5, 50,))
    #out_ = lam(dot)

    #out_ = dot
    model = Model(inputs=[query, doc], outputs=out_)
    '''
    l_s = Sequential()
    l_s.add(Embedding(2000, 50, input_length = 5))
    print l_s.output_shape
    l_r = Sequential()
    l_r.add(Embedding(2000, 50, input_length = 10))
    print l_s.output_shape

    model = Sequential()
    model.add(Merge([l_s, l_r], mode='dot', dot_axes=1))
    print model.output_shape
    '''
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
            metrics=['binary_crossentropy', 'accuracy'])
    val = model.predict([np.array([[1, 5]]),np.array([[2, 3, 4]])])
    print val
    print val.shape

    return model
matching()

