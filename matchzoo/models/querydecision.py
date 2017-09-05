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
                   'q_kernel_size', 'q_kernel_count',
                   'd_kernel_size', 'd_kernel_count',
                   'q_pool_size', 'd_pool_size', 'pair_feat_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[QueryDecision] parameter check wrong')
        print '[QueryDecision] init done'
        
    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        self.set_default('q_kernel_count', 32)
        self.set_default('q_kernel_size', 3)
        self.set_default('d_kernel_count', 32)
        self.set_default('d_kernel_size', 3)
        self.set_default('q_pool_size', 2)
        self.set_default('d_pool_size', 2)
        self.config.update(config)

    def build(self):
        def query_atten(x):
            att_x = x[0] * x[2]
            att_y = x[1] * (1. - x[2])
            return Concatenate(axis=1)([att_x, att_y])
        def query_noatten(x):
            return Concatenate(axis=1)([x[0], x[1]])
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[Input] query:\t%s' % str(query.get_shape().as_list())) 
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list())) 
        query_len = Input(name='query_len', shape=(1,))
        print('[Input] query_len:\t%s' % str(query_len.get_shape().as_list())) 
        pair_feats = Input(name='pair_feats', shape=(self.config['pair_feat_size'],))
        print('[Input] pair_feats:\t%s' % str(pair_feats.get_shape().as_list())) 
        query_feats = Input(name='query_feats', shape=(self.config['text1_maxlen'],))
        print('[Input] query_feats:\t%s' % str(query_feats.get_shape().as_list())) 

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)

        q_embed = embedding(query)
        print('[Embedding] query_embed:\t%s' % str(q_embed.get_shape().as_list())) 
        d_embed = embedding(doc)
        print('[Embedding] doc_embed:\t%s' % str(d_embed.get_shape().as_list())) 

        q_conv1 = Conv1D(
                self.config['q_kernel_count'], 
                self.config['q_kernel_size'], 
                padding='same', 
                activity_regularizer=regularizers.l2(0.01)
                ) (q_embed)
        print('[Conv1D] query_conv1:\t%s' % str(q_conv1.get_shape().as_list())) 
        d_conv1 = Conv1D(
                self.config['d_kernel_count'], 
                self.config['d_kernel_size'], 
                padding='same', 
                activity_regularizer=regularizers.l2(0.01)
                ) (d_embed)
        print('[Conv1D] doc_conv1:\t%s' % str(d_conv1.get_shape().as_list())) 

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        print('[MaxPooling1D] query_pool1:\t%s' % str(q_pool1.get_shape().as_list())) 
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)
        print('[MaxPooling1D] doc_pool1:\t%s' % str(d_pool1.get_shape().as_list())) 

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])
        print('[Concatenate] pool1:\t%s' % str(pool1.get_shape().as_list())) 

        pool1_flat = Reshape((-1,))(pool1)

        average = Lambda(lambda x: tf.reduce_mean(x, axis=1), output_shape=(1, self.config['embed_size'], ))(q_embed)
        print('[Lambda-Average] query_average:\t%s' % str(average.get_shape().as_list())) 
        #average = Lambda(lambda x: tf.reduce_min(x, axis=1), output_shape=(1, self.config['embed_size'], ))(q_embed)
        #average = Lambda(lambda x: tf.reduce_sum(x[0], axis=1)/x[1], output_shape=(1, self.config['embed_size'], ))([q_embed, query_len])
        average = Reshape((-1,))(average)

        #attr_feats = Concatenate(axis=1)([average, query_feats])
        #attr_feats = query_feats
        #print(attr_feats.get_shape().as_list())

        attr = Dense(1, activation='sigmoid', use_bias=False)(average)
        print('[Attention] query_attention:\t%s' % str(attr.get_shape().as_list())) 

        pool1_flat_d = Dense(1)(pool1_flat)
        print('[Dense] Dense of Concatenate:\t%s' % str(pool1_flat_d.get_shape().as_list())) 
        pair_feats_d = Dense(1)(pair_feats)
        print('[Dense] Dense of pair_feats:\t%s' % str(pair_feats_d.get_shape().as_list())) 

        feat = Lambda(query_atten)([pool1_flat_d, pair_feats_d, attr])
        print('[Attention] Attention Matching:\t%s' % str(feat.get_shape().as_list())) 
        #feat = Lambda(query_noatten)([pool1_flat, pair_feats])

        #out_ = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims=True), output_shape=(1, ))(feat)
        #out_ = Dense(1)(feat)
        out_ = Dense(1)(pool1_flat)
        #out_ = Dense(1)(pair_feats)
        print('[Dense] Matching Score:\t%s' % str(out_.get_shape().as_list())) 

        model = Model(inputs=[query, doc, query_len, pair_feats, query_feats], outputs=out_)
        return model
