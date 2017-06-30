
# -*- coding=utf-8 -*-
import keras
import keras.backend as K

def drmm(config):
    query = Input(name='query', shape=(config['text1_maxlen'],))
    doc = Input(name='doc', shape=(config['text1_maxlen'], config['hist_size']))

    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[config['embed']], trainable = False)

    q_embed = embedding(query)
    print K.int_shape(q_embed)
    q_reshape = Reshape((config['embed_size'], 1))(q_embed)
    print K.int_shape(q_reshape)
    q_dense = Dense(1)(q_reshape)
    print K.int_shape(q_dense)
    q_w = Reshape((config['text_maxlen'],))(q_dense)
    print K.int_shape(q_w)
    q_w = Activation('softmax', axis=1)(q_w)
    print K.int_shape(q_w)
    hist = Reshape((config['hist_size'],))(hist)
    print K.int_shape(hist)
    hist = Dense((config['drmm_mlp_0'],))(hist)
    hist = Activation('tanh')(hist)
    print K.int_shape(hist)
    hist = Dense((config['drmm_mlp_1'],))(hist)
    hist = Activation('tanh')(hist)
    print K.int_shape(hist)
    hist = Reshape((config['text1_maxlen'],))(hist)

    out_ = Merge('')(q_w, hist)

    model = Model(inputs=[query, doc], outputs=[out_])
    return model
