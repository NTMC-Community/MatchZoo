# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from Match import *
from utility import *

class MVLSTM(BasicModel):
    def __init__(self, config):
        super(MVLSTM, self).__init__(config)
        self.__name = 'MVLSTM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MVLSTM] parameter check wrong')
        print '[MVLSTM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(query.get_shape().as_list()), show_memory_use()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[layer]: Input\t[shape]: %s] \n%s' % (str(doc.get_shape().as_list()), show_memory_use()))
        #dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(q_embed.get_shape().as_list()), show_memory_use()))
        d_embed = embedding(doc)
        print('[layer]: Embedding\t[shape]: %s] \n%s' % (str(d_embed.get_shape().as_list()), show_memory_use()))

        q_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True))(q_embed)
        print('[layer]: Bi-LSTM\t[shape]: %s] \n%s' % (str(q_rep.get_shape().as_list()), show_memory_use()))

        d_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True))(d_embed)
        print('[layer]: Bi-LSTM\t[shape]: %s] \n%s' % (str(d_rep.get_shape().as_list()), show_memory_use()))

        cross = Match(match_type='dot')([q_rep, d_rep])
        #cross = Dot(axes=[2, 2])([q_embed, d_embed])
        print('[layer]: Match\t[shape]: %s] \n%s' % (str(cross.get_shape().as_list()), show_memory_use()))

        cross_reshape = Reshape((-1, ))(cross)
        print('[layer]: Reshape\t[shape]: %s] \n%s' % (str(cross_reshape.get_shape().as_list()), show_memory_use()))

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=False)[0])(cross_reshape)
        print('[layer]: Lambda-Topk\t[shape]: %s] \n%s' % (str(mm_k.get_shape().as_list()), show_memory_use()))

        #pool1_flat = Flatten()(mm_k)
        #print('[Flatten] pool1_flat:\t%s' % str(pool1_flat.get_shape().as_list()))

        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(mm_k)
        print('[layer]: Dropout\t[shape]: %s] \n%s' % (str(pool1_flat_drop.get_shape().as_list()), show_memory_use()))

        out_ = Dense(1)(pool1_flat_drop)
        print('[layer]: Dense\t[shape]: %s] \n%s' % (str(out_.get_shape().as_list()), show_memory_use()))

        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
