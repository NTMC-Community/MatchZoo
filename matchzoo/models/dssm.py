
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax


class DSSM(BasicModel):
    def __init__(self, config):
        super(DSSM, self).__init__(config)
        self.__name = 'DSSM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 
                   'num_layers', 'hidden_sizes']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print '[DSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        self.set_default('num_layers', 2)
        self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen']))

        def mlp_work(input_dim):
            seq = Sequential()
            seq.add(SparseFullyConnectedLayer(self.config['hidden_layers'][0], input_dim=input_dim, activation='relu'))
            for i in range(self.config['num_layers']-1):
                seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
            return seq
            
        assert self.config['text1_maxlen'] == self.config['text2_maxlen']
        mlp = mlp_work(self.config['text1_maxlen'])
        rq = mlp(query)
        rd = mlp(doc)
        out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        #out_ = Dot( axes= [1, 1])([rq, rd])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
