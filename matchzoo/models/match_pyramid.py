# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

def match_pyramid(config):
    query = Input(name='query', shape=(config['data1_maxlen'],))
    #print K.int_shape(query)
    doc = Input(name='doc', shape=(config['data2_maxlen'],))
    #print K.int_shape(doc)
    embedding = Embedding(config['vocab_size'], config['embed_size'])
    q_embed = embedding(query)
    d_embed = embedding(doc)
    #print K.int_shape(q_embed)
    #print K.int_shape(d_embed)

    z = Dot(axes=[2, 2])([q_embed, d_embed])
    #z = Dropout(0.2)(z)
    #print K.int_shape(z)
    z = Reshape((config['data1_maxlen'], config['data2_maxlen'], 1))(z)
    #print K.int_shape(z)
    conv2d_0 = Conv2D(8, (3, 3), padding='same', activation='relu')
    conv2d_1 = Conv2D(8, (3, 3), padding='same', activation='relu')
    mpool = MaxPooling2D(pool_size=(3,3), strides=(3, 3), padding='same')
    z = conv2d_0(z)
    #print 'conv 1: ', K.int_shape(z)
    z = mpool(z)
    #print 'mpool 1: ', K.int_shape(z)
    z = conv2d_1(z)
    #print 'conv 2: ', K.int_shape(z)
    z = mpool(z)
    #print 'mpool 2: ', K.int_shape(z)
    z = Flatten()(z)
    #print K.int_shape(z)
    out_ = Dense(1)(z)
    #print K.int_shape(out_)
    #loss = merge([out_], mode=rank_hinge_loss, name='loss', output_shape=(1,))

    model = Model(inputs=[query, doc], outputs=out_)
    return model
