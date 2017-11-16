# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from model import BasicModel
from keras.activations import softmax

import sys
sys.path.append('../matchzoo/utils/')
from utility import *

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
        print '[DRMM_TKS] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('topk', 20)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(q_embed.get_shape().as_list()), show_memory_use()))
        d_embed = embedding(doc)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(d_embed.get_shape().as_list()), show_memory_use()))
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        print('[layer]: Dot\t[shape]: %s] \n%s' % (str(mm.get_shape().as_list()), show_memory_use()))

        # compute term gating
        w_g = Dense(1)(q_embed)
        print('[layer]: Dense\t[shape]: %s] \n%s' % (str(w_g.get_shape().as_list()), show_memory_use()))
        g = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(w_g)
        print('[layer]: Lambda-softmax\t[shape]: %s] \n%s' % (str(g.get_shape().as_list()), show_memory_use()))
        g = Reshape((self.config['text1_maxlen'],))(g)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(g.get_shape().as_list()), show_memory_use()))

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(mm)
        print('[layer]: Lambda-Topk\t[shape]: %s] \n%s' % (str(mm_k.get_shape().as_list()), show_memory_use()))

        for i in range(self.config['num_layers']):
            mm_k = Dense(self.config['hidden_sizes'][i], activation='softplus', kernel_initializer='he_uniform', bias_initializer='zeros')(mm_k)
            print('[layer]: Dense\t[shape]: %s] \n%s' % (str(mm_k.get_shape().as_list()), show_memory_use()))

        mm_reshape = Reshape((self.config['text1_maxlen'],))(mm_k)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(mm_reshape.get_shape().as_list()), show_memory_use()))

        mean = Dot(axes=[1, 1])([mm_reshape, g])
        print('[layer]: Dot\t[shape]: %s] \n%s' % (str(mean.get_shape().as_list()), show_memory_use()))

        out_ = Reshape((1,))(mean)
        print('[layer]: Dense\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        model = Model(inputs=[query, doc], outputs=out_)
        return model
