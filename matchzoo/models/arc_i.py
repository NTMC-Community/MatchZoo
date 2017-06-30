# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

def arc_i(config):
    query = Input(name='query', shape=(config['text1_maxlen'],))
    #print K.int_shape(query)
    doc = Input(name='doc', shape=(config['text2_maxlen'],))

    #print K.int_shape(doc)
    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[config['embed']], trainable = False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    conv = Convolution1D(filters = 64, kernel_size=3, padding='same', activation='relu',strides=1)
    mpool = MaxPooling1D(pool_size = 2, padding = 'same')

    q_conv = conv(q_embed)
    q_mp = mpool(q_conv)
    d_conv = conv(d_embed)
    d_mp = mpool(d_conv)
    q_z = Flatten()(q_mp)
    d_z = Flatten()(d_mp)
    z = Concatenate()([q_z, d_z])

    #print K.int_shape(q_embed)
    #print K.int_shape(z)
    out_ = Dense(1)(z)
    #print K.int_shape(out_)
    #loss = merge([out_], mode=rank_hinge_loss, name='loss', output_shape=(1,))

    model = Model(inputs=[query, doc], outputs=out_)
    return model
