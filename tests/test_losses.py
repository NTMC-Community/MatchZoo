# -*- coding: utf-8 -*=
import numpy as np

import os
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

sys.path.append('/home/fanyixing/MatchZoo/matchzoo/losses/')
sys.path.append('/home/fanyixing/MatchZoo/matchzoo/metrics/')
sys.path.append('/home/fanyixing/MatchZoo/matchzoo/utils/')
sys.path.append('/home/fanyixing/MatchZoo/matchzoo/models/')
from rank_io import *
from rank_data_generator import *
from rank_losses import *
from rank_evaluations import *
from match_pyramid import *


MAX_Q_LEN = 5
MAX_D_LEN = 50

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding

if __name__ == '__main__':
    #base_dir = '/home/fanyixing/MatchZoo/sample_data/'
    base_dir = '/home/fanyixing/SouGou/origin_cut_all/'
    query_file = base_dir + 'query.txt'
    doc_file = base_dir + 'doc.txt'
    #embed_file = base_dir + 'embed_sogou_d50_norm'
    train_rel_file = base_dir + 'relation.train.fold0.txt'
    valid_rel_file = base_dir + 'relation.valid.fold0.txt'
    test_rel_file = base_dir + 'relation.test.fold0.txt'

    queries = read_data(query_file)
    print 'Total queries : %d ...'%(len(queries))
    docs =  read_data(doc_file)
    print 'Total docs : %d ...'%(len(docs))
    #embed = read_embedding(embed_file)

    config = {}
    config['vocab_size'] = 360287 + 1
    #config['vocab_size'] = 26075 + 1
    config['embed_size'] = 100
    config['data1_maxlen'] = 5
    config['data2_maxlen'] = 50
    config['batch_size'] = 100
    #config['fill_word'] = 26075
    config['fill_word'] = 360287
    config['learning_rate'] = 0.0001
    config['epochs'] = 1 

    pair_gen = PairGenerator(train_rel_file, data1=queries, data2=docs, config=config)
    list_gen = ListGenerator(test_rel_file, data1=queries, data2=docs, config=config)
    #x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls = list_gen.get_all_data()

    model = match_pyramid(config)
    #for (x1, x1_len, x2, x2_len, y_true) in list_gen.get_batch:
    #   print(y_true)
    #eval_map = MAP_eval(validation_data = list_gen, rel_threshold=0)
    #eval_map = MAP_eval(x1_ls, x2_ls, y_ls, rel_threshold=0)
    rank_eval = rank_eval(rel_threshold = 0.)

    model.compile(optimizer=Adam(lr=config['learning_rate']), loss=rank_hinge_loss)

    for k in range(config['epochs']):
        num_batch = pair_gen.num_pairs / config['batch_size']
        for i in range(num_batch):
            x1, x1_len, x2, x2_len, y = pair_gen.get_batch
            model.fit({'query':x1, 'doc':x2}, y, batch_size=config['batch_size']*2,
                    epochs = 1,
                    verbose = 1
                    ) #callbacks=[eval_map])
            if i % 100 == 0:
                res = [0., 0., 0.] 
                num_valid = 0
                for (x1, x1_len, x2, x2_len, y_true) in list_gen.get_batch:
                    y_pred = model.predict({'query': x1, 'doc': x2})
                    curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=['map', 'ndcg@3', 'ndcg@5'])
                    res[0] += curr_res['map']
                    res[1] += curr_res['ndcg@3']
                    res[2] += curr_res['ndcg@5']
                    num_valid += 1
                res[0] /= num_valid
                res[1] /= num_valid
                res[2] /= num_valid
                print 'epoch: %d, batch : %d , map: %f, ndcg@3: %f, ndcg@5: %f ...'%(k, i, res[0], res[1], res[2])
                sys.stdout.flush()
                list_gen.reset
    



