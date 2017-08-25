
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Convolution1D, MaxPooling1D, Reshape, Dot
from keras.activations import softmax

from model import BasicModel

class CDSSM(BasicModel):
    def __init__(self, config):
        super(CDSSM, self).__init__(config)
        self.__name = 'CDSSM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 
                   'vocab_size', 'embed_size',
                   'filters', 'kernel_size', 'hidden_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[CDSSM] parameter check wrong')
        print '[CDSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        self.set_default('num_layers', 2)
        self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))

        wordhashing = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = wordhashing(query)
        d_embed = wordhashing(doc)
        conv1d = Convolution1D(self.config['filters'], self.config['kernel_size'], padding='same', activation='relu')
        q_conv = conv1d(q_embed)
        d_conv = conv1d(d_embed)
        q_pool = MaxPooling1D(self.config['text1_maxlen'])(q_conv)
        d_pool = MaxPooling1D(self.config['text2_maxlen'])(d_conv)
        mlp = Dense(self.config['hidden_size'], activation='tanh')
        rq = mlp(query)
        rd = mlp(doc)
        #out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
