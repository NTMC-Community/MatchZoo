# -*- coding: utf-8 -*=
import numpy as np

import os
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

sys.path.append('/home/fanyixing/MatchZoo/matchzoo/losses')
sys.path.append('/home/fanyixing/MatchZoo/matchzoo/metrics')
from rank_losses import *
from rank_evaluations import *


MAX_Q_LEN = 5
MAX_D_LEN = 50

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding

def matching(lr=0.1):
    query = Input(name='query', shape=(5,))
    print K.int_shape(query)
    doc = Input(name='doc', shape=(30,))
    print K.int_shape(doc)
    embedding = Embedding(2000, 10)
    q_embed = embedding(query)
    d_embed = embedding(doc)
    print K.int_shape(q_embed)
    print K.int_shape(d_embed)

    #dot = K.batch_dot(q_embed, d_embed, axes=[2, 2])
    #dot = Merge([q_embed, d_embed], mode='dot', dot_axes=1)
    z = Dot(axes=[2, 2])([q_embed, d_embed])
    z = Dropout(0.2)(z)
    print K.int_shape(z)
    z = Reshape((5, 30, 1))(z)
    print K.int_shape(z)
    conv2d_0 = Conv2D(8, (3, 3), padding='same', activation='relu')
    conv2d_1 = Conv2D(8, (3, 3), padding='same', activation='relu')
    mpool = MaxPooling2D(pool_size=(3,3), strides=(3, 3), padding='same')
    z = conv2d_0(z)
    print 'conv 1: ', K.int_shape(z)
    z = mpool(z)
    print 'mpool 1: ', K.int_shape(z)
    z = conv2d_1(z)
    print 'conv 2: ', K.int_shape(z)
    z = mpool(z)
    print 'mpool 2: ', K.int_shape(z)
    z = Flatten()(z)
    print K.int_shape(z)
    out_ = Dense(1)(z)
    print K.int_shape(out_)
    #loss = merge([out_], mode=rank_hinge_loss, name='loss', output_shape=(1,))

    model = Model(inputs=[query, doc], outputs=out_)

    model.compile(optimizer=Adam(lr=lr), loss=rank_hinge_loss,
            metrics=[eval_map])
    #val = model.predict([np.array([[1, 5]]),np.array([[2, 3, 4]])])
    #print val
    #print val.shape
    return model
matching()

