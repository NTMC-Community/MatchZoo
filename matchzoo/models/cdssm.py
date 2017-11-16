
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Convolution1D, MaxPooling1D, Reshape, Dot
from keras.activations import softmax

from model import BasicModel
import sys
sys.path.append('../matchzoo/utils/')
from utility import *


class CDSSM(BasicModel):
    def __init__(self, config):
        super(CDSSM, self).__init__(config)
        self.__name = 'CDSSM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'vocab_size', 'embed_size',
                   'filters', 'kernel_size', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[CDSSM] parameter check wrong')
        print '[CDSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.config['hidden_sizes'])
            assert num_hidden_layers > 0
            if num_hidden_layers == 1:
                seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,)))
            else:
                seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
                for i in range(num_hidden_layers - 2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
                seq.add(Dense(self.config['hidden_sizes'][num_hidden_layers-1]))
            return seq
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))

        wordhashing = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = wordhashing(query)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(q_embed.get_shape().as_list()), show_memory_use()))
        d_embed = wordhashing(doc)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(d_embed.get_shape().as_list()), show_memory_use()))
        conv1d = Convolution1D(self.config['filters'], self.config['kernel_size'], padding='same', activation='relu')
        q_conv = conv1d(q_embed)
        print('[layer]: Conv1D\t[shape]: %s] \n%s' % (str(q_conv.get_shape().as_list()), show_memory_use()))
        d_conv = conv1d(d_embed)
        print('[layer]: Conv1D\t[shape]: %s] \n%s' % (str(d_conv.get_shape().as_list()), show_memory_use()))
        q_pool = MaxPooling1D(self.config['text1_maxlen'])(q_conv)
        print('[layer]: MaxPooling1D\t[shape]: %s] \n%s' % (str(q_pool.get_shape().as_list()), show_memory_use()))
        q_pool_re = Reshape((-1,))(q_pool)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(q_pool_re.get_shape().as_list()), show_memory_use()))
        d_pool = MaxPooling1D(self.config['text2_maxlen'])(d_conv)
        print('[layer]: MaxPooling1D\t[shape]: %s] \n%s' % (str(d_pool.get_shape().as_list()), show_memory_use()))
        d_pool_re = Reshape((-1,))(d_pool)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(d_pool_re.get_shape().as_list()), show_memory_use()))


        mlp = mlp_work(self.config['embed_size'])

        rq = mlp(q_pool_re)
        print('[layer]: MLP\t[shape]: %s] \n%s' % (str(rq.get_shape().as_list()), show_memory_use()))
        rd = mlp(d_pool_re)
        print('[layer]: MLP\t[shape]: %s] \n%s' % (str(rd.get_shape().as_list()), show_memory_use()))
        #out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        print('[layer]: Dot\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
