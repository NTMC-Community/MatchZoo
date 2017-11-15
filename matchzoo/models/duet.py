# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import *
#from keras.layers import Reshape, Embedding,Merge, Dot, Lambda
from keras.optimizers import Adam
from model import BasicModel
import tensorflow as tf

import sys
sys.path.append('../matchzoo/layers/')
from Match import *

class Duet(BasicModel):
    def __init__(self, config):
        super(Duet, self).__init__(config)
        self.__name = 'Duet'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen'
                   ]
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[Duet] parameter check wrong')
        print '[Duet] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        #self.set_default('q_kernel_count', 32)
        #self.set_default('q_kernel_size', 3)
        #self.set_default('d_kernel_count', 32)
        #self.set_default('d_kernel_size', 3)
        #self.set_default('q_pool_size', 2)
        #self.set_default('d_pool_size', 2)
        self.config.update(config)

    def build(self):
        def xor_match(x):
            t1 = x[0]
            t2 = x[1]
            t1_shape = t1.get_shape()
            t2_shape = t2.get_shape()
            t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
            t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
            out_bool = K.tf.equal(t1_expand, t2_expand)
            out = K.tf.cast(out_bool, K.tf.float32)
            return out
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[Input] query:\t%s' % str(query.get_shape().as_list()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list()))

        xor = Lambda(xor_match)([query, doc])
        print('[Lambda: XOR] xor:\t%s' % str(xor.get_shape().as_list()))

        xor_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(xor)
        print('[Reshape: [5,5]] xor_reshape:\t%s' % str(xor_reshape.get_shape().as_list()))

        conv2d = Conv2D(20, [5, 5], padding='same', activation='relu')(xor_reshape)
        print('[Conv2D: [5,5]] conv2d:\t%s' % str(conv2d.get_shape().as_list()))

        pool2d = MaxPooling2D([2, 10], strides=[2, 10], padding='valid')(conv2d)
        print('[MaxPooling2D: [2,10]] pool2d:\t%s' % str(pool2d.get_shape().as_list()))


        pool1_flat = Reshape((-1,))(pool2d)
        print('[Reshape] pool1_flat:\t%s' % str(pool1_flat.get_shape().as_list()))

        out_ = Dense(1)(pool1_flat)
        #out_ = Dense(1)(pool1_flat)
        #out_ = Dense(1)(pair_feats)
        print('[Dense: 1] Matching Score:\t%s' % str(out_.get_shape().as_list()))

        model = Model(inputs=[query, doc], outputs=out_)
        return model
