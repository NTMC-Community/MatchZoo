# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
from model import BasicModel

import sys
sys.path.append('../matchzoo/utils/')
from utility import *

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
        print '[DRMM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('text1_maxlen', 5)
        self.set_default('hist_size', 60)
        self.config.update(config)

    def build(self):
        def tensor_product(x):
            a = x[0]
            b = x[1]
            y = K.batch_dot(a, b, axis=1)
            y = K.einsum('ijk, ikl->ijl', a, b)
            return y
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['text1_maxlen'], self.config['hist_size']))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = False)

        q_embed = embedding(query)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(q_embed.get_shape().as_list()), show_memory_use()))
        q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        print('[layer]: Dense\t[shape]: %s] \n%s' % (str(q_w.get_shape().as_list()), show_memory_use()))
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(q_w)
        print('[layer]: Lambda-softmax\t[shape]: %s] \n%s' % (str(q_w.get_shape().as_list()), show_memory_use()))
        z = doc
        for i in range(self.config['num_layers']):
            z = Dense(self.config['hidden_sizes'][i], kernel_initializer=self.initializer_fc)(z)
            z = Activation('tanh')(z)
            print('[layer]: Dense\t[shape]: %s] \n%s' % (str(z.get_shape().as_list()), show_memory_use()))
        z = Permute((2, 1))(z)
        print('[layer]: Permute\t[shape]: %s] \n%s' % (str(z.get_shape().as_list()), show_memory_use()))
        z = Reshape((self.config['text1_maxlen'],))(z)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(z.get_shape().as_list()), show_memory_use()))
        q_w = Reshape((self.config['text1_maxlen'],))(q_w)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(q_w.get_shape().as_list()), show_memory_use()))

        out_ = Dot( axes= [1, 1])([z, q_w])
        print('[layer]: Dot\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
