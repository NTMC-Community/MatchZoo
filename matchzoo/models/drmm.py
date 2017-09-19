# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
from model import BasicModel

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
        print('[Input] query:\t%s' % str(query.get_shape().as_list())) 
        doc = Input(name='doc', shape=(self.config['text1_maxlen'], self.config['hist_size']))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list())) 

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = False)

        q_embed = embedding(query)
        print('[Embedding] q_embed:\t%s' % str(q_embed.get_shape().as_list())) 
        q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        print('[Dense] q_gate:\t%s' % str(q_w.get_shape().as_list())) 
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(q_w)
        print('[Softmax] q_gate:\t%s' % str(q_w.get_shape().as_list())) 
        z = doc
        for i in range(self.config['num_layers']):
            z = Dense(self.config['hidden_sizes'][i], kernel_initializer=self.initializer_fc)(z)
            z = Activation('tanh')(z)
            print('[Dense] z (full connection):\t%s' % str(z.get_shape().as_list())) 
        z = Permute((2, 1))(z)
        z = Reshape((self.config['text1_maxlen'],))(z)
        print('[Reshape] z (matching) :\t%s' % str(z.get_shape().as_list())) 
        q_w = Reshape((self.config['text1_maxlen'],))(q_w)
        print('[Reshape] q_w (gating) :\t%s' % str(q_w.get_shape().as_list())) 

        out_ = Dot( axes= [1, 1])([z, q_w])
        print('[Dot] out_ :\t%s' % str(out_.get_shape().as_list())) 

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
