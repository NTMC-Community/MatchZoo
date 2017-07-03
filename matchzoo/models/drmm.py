
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax

def check(config):
    ''' check config information for building
    '''
    def default_config(config):
        return config

    config = default_config(config)
    check_list = [ 'text1_maxlen', 'hist_size',
                   'embed', 'embed_size', 'vocab_size',
                   'num_layers', 'hidden_sizes']
    for e in check_list:
        if e not in config:
            print '[Model] Error %s not in config' % e
            return False
    return True
def build(config):
    def tensor_product(x):
        a = x[0]
        b = x[1]
        y = K.batch_dot(a, b, axis=1)
        y = K.einsum('ijk, ikl->ijl', a, b)
        return y
    query = Input(name='query', shape=(config['text1_maxlen'],))
    doc = Input(name='doc', shape=(config['text1_maxlen'], config['hist_size']))

    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[config['embed']], trainable = False)

    q_embed = embedding(query)
    print 'q_embed:\t', K.int_shape(q_embed)

    #q_reshape = Reshape((-1, config['embed_size'],))(q_embed)
    #print K.int_shape(q_reshape)
    q_w = Dense(1)(q_embed)
    print 'q_w:\t', K.int_shape(q_w)
    #q_w = Reshape((config['text1_maxlen'],))(q_dense)
    print K.int_shape(q_w)
    #q_w = softmax(q_w, axis=1)
    q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(config['text1_maxlen'], ))(q_w)
    #q_w = Activation('softmax', axis=1)(q_w)
    print K.int_shape(q_w)
    #z = Reshape((config['hist_size'],))(hist)
    z = doc
    print K.int_shape(z)
    for i in range(config['num_layers']):
        z = Dense(config['hidden_sizes'][i])(z)
        print 'layer %d: '%(i), K.int_shape(z)
        z = Activation('tanh')(z)
    #hist = Reshape((config['text1_maxlen'],))(hist)
    z = Permute((2, 1))(z)
    print 'z:\t', K.int_shape(z)
    z = Reshape((config['text1_maxlen'],))(z)
    print 'z:\t', K.int_shape(z)
    q_w = Reshape((config['text1_maxlen'],))(q_w)
    print 'q_w:\t', K.int_shape(q_w)

    out_ = Dot( axes= [1, 1])([z, q_w])
    #out_ = Merge('sum')(out_)
    print K.int_shape(out_)

    model = Model(inputs=[query, doc], outputs=[out_])
    return model
