# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from model import BasicModel
from keras.activations import softmax

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
        print('[Input] query:\t%s' % str(query.get_shape().as_list())) 
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        print('[Input] doc:\t%s' % str(doc.get_shape().as_list())) 

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        print('[Embedding] query_embed:\t%s' % str(q_embed.get_shape().as_list())) 
        d_embed = embedding(doc)
        print('[Embedding] doc_embed:\t%s' % str(d_embed.get_shape().as_list())) 
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        print('[Dot] mm:\t%s' % str(mm.get_shape().as_list())) 
        
        # compute term gating
        w_g = Dense(1)(q_embed) 
        print('[Dense] w_g:\t%s' % str(w_g.get_shape().as_list())) 
        g = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(w_g) 
        print('[Lambda: softmax] g:\t%s' % str(g.get_shape().as_list())) 
        g = Reshape((self.config['text1_maxlen'],))(g)
        print('[Reshape] g:\t%s' % str(g.get_shape().as_list())) 

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(mm)
        print('[Lambda: top_k %d] mm_k:\t%s' % (self.config['topk'], str(mm_k.get_shape().as_list()))) 

        for i in range(self.config['num_layers']):
            mm_k = Dense(self.config['hidden_sizes'][i], activation='softplus', kernel_initializer='he_uniform', bias_initializer='zeros')(mm_k)
            print('[Dense (%d): %d] mm_k:\t%s' % (i, self.config['hidden_sizes'][i], str(mm_k.get_shape().as_list())))

        mm_reshape = Reshape((self.config['text1_maxlen'],))(mm_k)
        print('[Reshape] mm_reshape :\t%s' % str(mm_reshape.get_shape().as_list())) 

        mean = Dot(axes=[1, 1])([mm_reshape, g])
        print('[Dot] mean :\t%s' % str(mean.get_shape().as_list())) 

        #out_ = Reshape((1,))(mean)
        out_ = Dense(2, activation='softmax')(mean)
        print('[Reshape] out_ :\t%s' % str(out_.get_shape().as_list())) 

        model = Model(inputs=[query, doc], outputs=out_)
        return model
