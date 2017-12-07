# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from model import BasicModel
from keras.activations import softmax
from utils.utility import *

class DRMM_TKS(BasicModel):
    """DRMM_TKS model, this is a variant version of DRMM, which applied topk pooling in the matching matrix.

    Firstly, embed queries into embedding vector named 'q_embed' and 'd_embed' respectively.
    Secondly, computing 'q_embed' and 'd_embed' with element-wise multiplication,
    Thirdly, computing output of upper layer with dense layer operation,
    then take softmax operation on the output of this layer named 'g' and
    find the k largest entries named 'mm_k'.
    Fourth, input 'mm_k' into hidden layers, with specified length of layers and activation function.
    Lastly, compute 'g' and 'mm_k' with element-wise multiplication.

    # Returns
	Score list between queries and documents.
    """
    def __init__(self, config):
        super(DRMM_TKS, self).__init__(config)
        self.__name = 'DRMM_TKS'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'topk', 'num_layers', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[DRMM_TKS] parameter check wrong')
        print('[DRMM_TKS] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('topk', 20)
        self.set_default('dropout_rate', 0.)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        show_layer_info('Dot', mm)

        # compute term gating
        w_g = Dense(1)(q_embed)
        show_layer_info('Dense', w_g)
        g = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(w_g)
        show_layer_info('Lambda-softmax', g)
        g = Reshape((self.config['text1_maxlen'],))(g)
        show_layer_info('Reshape', g)

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(mm)
        show_layer_info('Lambda-topk', mm_k)

        for i in range(self.config['num_layers']):
            mm_k = Dense(self.config['hidden_sizes'][i], activation='softplus', kernel_initializer='he_uniform', bias_initializer='zeros')(mm_k)
            show_layer_info('Dense', mm_k)

        mm_k_dropout = Dropout(rate=self.config['dropout_rate'])(mm_k)
        show_layer_info('Dropout', mm_k_dropout)

        mm_reshape = Reshape((self.config['text1_maxlen'],))(mm_k_dropout)
        show_layer_info('Reshape', mm_reshape)

        mean = Dot(axes=[1, 1])([mm_reshape, g])
        show_layer_info('Dot', mean)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(mean)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Reshape((1,))(mean)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
