# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.optimizers import Adam
from model import BasicModel
from layers.DynamicMaxPooling import *
from layers.BiLSTM import BiLSTM
from layers.MultiPerspectiveMatch import MultiPerspectiveMatch
#from layers.Attention import MultiPerspectiveAttention
from layers.SequenceMask import SequenceMask
from utils.utility import *


class BiMPM(BasicModel):
    """implementation of Bilateral Multi-Perspective Matching
        https://arxiv.org/pdf/1702.03814.pdf
    """
    def __init__(self, config):
        super(BiMPM, self).__init__(config)
        self.__name = 'BiMPM'
        self.check_list = ['text1_maxlen', 'text2_maxlen',
                           'embed', 'embed_size', 'vocab_size',
                            'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[BiMPM] parameter check wrong')
        print('[MatchPyramid] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        query_len = Input(name='query_len', shape=(1,))
        show_layer_info('Input', query_len)
        doc_len = Input(name='doc_len', shape=(1,))
        show_layer_info('Input', doc_len)

        q_mask = SequenceMask(self.config['text1_maxlen'])(query_len)
        show_layer_info('SequenceMask', q_mask)
        d_mask = SequenceMask(self.config['text2_maxlen'])(doc_len)
        show_layer_info('SequenceMask', d_mask)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        bilstm = BiLSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'])
        q_outs, q_out = bilstm(q_embed)
        show_layer_info("Bidirectional-LSTM", q_outs)
        d_outs, d_out = bilstm(d_embed)
        show_layer_info("Bidirectional-LSTM", d_outs)

        match = MultiPerspectiveMatch(self.config['channel'])
        q_match = match([d_outs, d_out, d_mask, q_outs, q_out, q_mask])
        show_layer_info("MultiPerspectiveMatch", q_match)
        d_match = match([q_outs, q_out, q_mask, d_outs, d_out, d_mask])
        show_layer_info("MultiPerspectiveMatch", d_match)

        aggre = BiLSTM(self.config['aggre_size'], dropout=self.config['dropout_rate'])
        q_outs, q_out = aggre(q_match)
        show_layer_info("Aggregation", q_outs)
        d_outs, d_out = aggre(d_match)
        show_layer_info("Aggregation", d_outs)

        flat = Concatenate(axis=1)([q_out, d_out])
        flat = Highway()(flat)
        show_layer_info("Highway", flat)

        flat_drop = Dropout(rate=self.config['dropout_rate'])(flat)
        show_layer_info('Dropout', flat_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc, query_len, doc_len], outputs=out_)
        return model
