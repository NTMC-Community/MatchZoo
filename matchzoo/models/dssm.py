
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
    check_list = [ 'text1_maxlen', 'text2_maxlen',
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
    doc = Input(name='doc', shape=(config['text2_maxlen']))

    def mlp_work(input_dim):
        seq = Sequential()
        seq.add(SparseFullyConnectedLayer(config['hidden_layers'][0], input_dim=input_dim, activation='relu'))
        for i in range(config['num_layers']-1):
            seq.add(Dense(config['hidden_sizes'][i+1], activation='relu'))
        return seq
        
    assert config['text1_maxlen'] == config['text2_maxlen']
    mlp = mlp_work(config['text1_maxlen'])
    rq = mlp(query)
    rd = mlp(doc)
    out_ = Merge([rq, rd], mode='cos', dot_axis=1)
    #out_ = Dot( axes= [1, 1])([rq, rd])

    model = Model(inputs=[query, doc], outputs=[out_])
    return model
