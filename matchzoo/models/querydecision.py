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

class QueryDecision(BasicModel):
    def __init__(self, config):
        super(QueryDecision, self).__init__(config)
        self.__name = 'QueryDecision'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'q_pool_size', 'd_pool_size', 'pair_feat_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[QueryDecision] parameter check wrong')
        print '[QueryDecision] init done'
        
    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', 3)
        self.set_default('q_pool_size', 2)
        self.set_default('d_pool_size', 2)
        self.config.update(config)

    def build(self):
        def query_atten(x):
            att_x = x[0] * x[2]
            print(att_x.get_shape().as_list())
            att_y = x[1] * (1. - x[2])
            print(att_y.get_shape().as_list())
            return Concatenate(axis=1)([att_x, att_y])
        def query_noatten(x):
            return Concatenate(axis=1)([x[0], x[1]])
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        query_len = Input(name='query_len', shape=(1,))
        pair_feats = Input(name='pair_feats', shape=(self.config['pair_feat_size'],))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)

        q_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activity_regularizer=regularizers.l2(0.01)) (q_embed)
        d_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activity_regularizer=regularizers.l2(0.01)) (d_embed)

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])

        #pool1_flat = Flatten()(pool1)
        pool1_flat = Reshape((-1,))(pool1)

        #average = Lambda(lambda x: tf.reduce_mean(x, axis=1), output_shape=(1, self.config['embed_size'], ))(q_embed)
        #average = Lambda(lambda x: tf.reduce_min(x, axis=1), output_shape=(1, self.config['embed_size'], ))(q_embed)
        average = Lambda(lambda x: tf.reduce_sum(x[0], axis=1)/x[1], output_shape=(1, self.config['embed_size'], ))([q_embed, query_len])

        print(average.get_shape().as_list())
        attr = Dense(1, activation='sigmoid', use_bias=False)(average)
        attr = Reshape((1,))(attr)
        print(attr.get_shape().as_list())

        pool1_flat_d = Dense(1)(pool1_flat)
        print(pool1_flat_d.get_shape().as_list())
        pair_feats_d = Dense(1)(pair_feats)
        print(pair_feats_d.get_shape().as_list())

        feat = Lambda(query_atten)([pool1_flat_d, pair_feats_d, attr])
        #feat = Lambda(query_noatten)([pool1_flat, pair_feats])

        out_ = Dense(1)(feat)
        #out_ = Dense(1)(pool1_flat)
        #out_ = Dense(1)(pair_feats)

        model = Model(inputs=[query, doc, query_len, pair_feats], outputs=out_)
        return model
