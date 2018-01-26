# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Input, Embedding, Dense, Activation, Lambda, Dot
from keras.activations import softmax
from keras.initializers import Constant, RandomNormal
from model import BasicModel
from utils.utility import *

class KNRM(BasicModel):
    def __init__(self, config):
        super(KNRM, self).__init__(config)
        self._name = 'KNRM'
        self.check_list = [ 'text1_maxlen', 'kernel_num','sigma','exact_sigma',
                            'embed', 'embed_size', 'vocab_size']
        self.setup(config)
        if not self.check():
            raise TypeError('[KNRM] parameter check wrong')
        print '[KNRM] init done'

    def setup(self, config):
        self.set_default('kernel_num', 11)
        self.set_default('sigma', 0.1)
        self.set_default('exact_sigma', 0.001)
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)

    def build(self):
        def Kernel_layer(mu,sigma):
            def kernel(x):
                return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return Activation(kernel)

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.config['train_embed'])

        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        show_layer_info('Dot', mm)

        KM = []
        for i in range(self.config['kernel_num']):
            mu = 1. / (self.config['kernel_num'] - 1) + (2. * i) / (self.config['kernel_num'] - 1) - 1.0
            sigma = self.config['sigma']
            if mu > 1.0:
                sigma = self.config['exact_sigma']
                mu = 1.0
            mm_exp = Kernel_layer(mu, sigma)(mm)
            show_layer_info('Exponent of mm:', mm_exp)
            mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x,2))(mm_exp)
            show_layer_info('Sum of document', mm_doc_sum)
            mm_log = Activation(K.tf.log1p)(mm_doc_sum)
            show_layer_info('Logarithm of sum', mm_log)
            mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log)
            show_layer_info('Sum of all exponent', mm_sum)
            KM.append(mm_sum)


        Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)
        show_layer_info('Stack', Phi)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax', kernel_initializer=initializers.RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1, kernel_initializer=initializers.RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model

