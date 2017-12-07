# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
from model import BasicModel
from utils.utility import *

class DRMM(BasicModel):
    def __init__(self, config):
        super(DRMM, self).__init__(config)
        self._name = 'DRMM'
        self.check_list = [ 'text1_maxlen', 'hist_size',
                'embed', 'embed_size', 'vocab_size',
                'num_layers', 'hidden_sizes']
        self.setup(config)
        self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[DRMM] parameter check wrong')
        print('[DRMM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('text1_maxlen', 5)
        self.set_default('hist_size', 60)
        self.set_default('dropout_rate', 0.)
        self.config.update(config)

    def build(self):
        def tensor_product(x):
            a = x[0]
            b = x[1]
            y = K.batch_dot(a, b, axis=1)
            y = K.einsum('ijk, ikl->ijl', a, b)
            return y
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text1_maxlen'], self.config['hist_size']))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = False)

        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        show_layer_info('Dense', q_w)
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(q_w)
        show_layer_info('Lambda-softmax', q_w)
        z = doc
        z = Dropout(rate=self.config['dropout_rate'])(z)
        show_layer_info('Dropout', z)
        for i in range(self.config['num_layers']-1):
            z = Dense(self.config['hidden_sizes'][i], kernel_initializer=self.initializer_fc)(z)
            z = Activation('tanh')(z)
            show_layer_info('Dense', z)
        z = Dense(self.config['hidden_sizes'][self.config['num_layers']-1], kernel_initializer=self.initializer_fc)(z)
        show_layer_info('Dense', z)
        z = Permute((2, 1))(z)
        show_layer_info('Permute', z)
        z = Reshape((self.config['text1_maxlen'],))(z)
        show_layer_info('Reshape', z)
        q_w = Reshape((self.config['text1_maxlen'],))(q_w)
        show_layer_info('Reshape', q_w)

        out_ = Dot( axes= [1, 1])([z, q_w])
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
