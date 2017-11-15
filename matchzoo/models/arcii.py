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
#from DynamicMaxPooling import *
from Match import *

class ARCII(BasicModel):
    def __init__(self, config):
        super(ARCII, self).__init__(config)
        self.__name = 'ARCII'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   '1d_kernel_size', '1d_kernel_count',
                   'num_conv2d_layers','2d_kernel_sizes',
                   '2d_kernel_counts','2d_mpool_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCII] parameter check wrong')
        print '[ARCII] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('1d_kernel_count', 32)
        self.set_default('1d_kernel_size', 3)
        self.set_default('num_conv2d_layers', 2)
        self.set_default('2d_kernel_counts', [32, 32])
        self.set_default('2d_kernel_sizes', [3, 3])
        self.set_default('2d_mpool_sizes', [[3, 3], [3,3]])
        self.config.update(config)

    def build(self):
        def conv2d_work(input_dim):
            seq = Sequential()
            assert self.config['num_conv2d_layers'] > 0
            for i in range(self.config['num_conv2d_layers']):
                seq.add(Conv2D(self.config['2d_kernel_counts'][i], self.config['2d_kernel_sizes'][i], padding='same', activation='relu'))
                seq.add(MaxPooling2D(pool_size=(self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1])))
            return seq
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        print('[Input] query:\t%s' % str(query.get_shape().as_list()))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list()))
        #dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')
        #print('[Input] dpool_index:\t%s' % str(dpool_index.get_shape().as_list()))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        print('[Embedding] query_embed:\t%s' % str(q_embed.get_shape().as_list()))
        d_embed = embedding(doc)
        print('[Embedding] doc_embed:\t%s' % str(d_embed.get_shape().as_list()))

        q_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (q_embed)
        print('[Conv1D: (%d, %d)] query_conv1:\t%s' % (self.config['1d_kernel_count'], self.config['1d_kernel_size'], str(q_conv1.get_shape().as_list())))
        d_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (d_embed)
        print('[Conv1D: (%d, %d)] doc_conv1:\t%s' % (self.config['1d_kernel_count'], self.config['1d_kernel_size'], str(d_conv1.get_shape().as_list())))

        cross = Match(match_type='plus')([q_conv1, d_conv1])
        print('[Match: plus] cross:\t%s' % str(cross.get_shape().as_list()))

        z = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], -1))(cross)
        print('[Reshape] z:\t%s' % str(z.get_shape().as_list()))

        for i in range(self.config['num_conv2d_layers']):
            z = Conv2D(self.config['2d_kernel_counts'][i], self.config['2d_kernel_sizes'][i], padding='same', activation='relu')(z)
            print('[Conv2D %d: (%d, %d)] z:\t%s' % (i, self.config['2d_kernel_counts'][i], self.config['2d_kernel_sizes'][i], str(z.get_shape().as_list())))
            z = MaxPooling2D(pool_size=(self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1]))(z)
            print('[MaxPooling2D %d: (%d, %d)] z:\t%s' % (i, self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1], str(z.get_shape().as_list())))

        #dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])([conv2d, dpool_index])
        #print('[DynamicMaxPooling] dpool:\t%s' % str(dpool.get_shape().as_list()))

        pool1_flat = Flatten()(z)
        print('[Flatten] pool1_flat:\t%s' % str(pool1_flat.get_shape().as_list()))
        out_ = Dense(1)(pool1_flat)
        print('[Dense] out_:\t%s' % str(out_.get_shape().as_list()))

        model = Model(inputs=[query, doc], outputs=out_)
        return model
