# -*- coding: utf8 -*-
import numpy as np

import os
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam

sys.path.append('../matchzoo/losses/')
sys.path.append('../matchzoo/metrics/')
sys.path.append('../matchzoo/utils/')
sys.path.append('../matchzoo/models/')
from rank_io import *
from rank_data_generator import *
from rank_losses import *
from rank_evaluations import *
from arc_i import *


MAX_Q_LEN = 5
MAX_D_LEN = 50

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding

if __name__ == '__main__':
    #base_dir = '/home/fanyixing/MatchZoo/sample_data/'
    #base_dir = '/home/fanyixing/SouGou/origin_cut_all/'
    #query_file = base_dir + 'query.txt'
    #doc_file = base_dir + 'doc.txt'
    #embed_file = base_dir + 'embed_sogou_d50_norm'
    embed_file = '/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/embed_wiki-pdc_d50_norm'


    config = {}
    config['text1_corpus'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/qid_query.txt"
    config['text2_corpus'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/docid_doc.txt"
    config['relation_train'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/relation.train.fold1.txt"
    config['relation_test'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/relation.test.fold1.txt"
    #config['text1_corpus']  = '../sample_data/query.txt'
    #config['text2_corpus'] = '../sample_data/doc.txt'
    #config['relation_train'] = '../sample_data/relation.train.fold0.txt'
    #config['relation_train'] = '../sample_data/relation.test.fold0.txt'
    config['vocab_size'] = 193367 + 1
    #config['vocab_size'] = 26075 + 1
    config['embed_size'] = 50
    config['text1_maxlen'] = 5
    config['text2_maxlen'] = 50
    config['batch_size'] = 1
    #config['fill_word'] = 26075
    config['fill_word'] = 193367
    config['learning_rate'] = 0.0001
    config['num_epochs'] = 1 

    query_file = config['text1_corpus']
    doc_file = config['text2_corpus']
    embed = read_embedding(embed_file)
    print len(embed)
    print config['vocab_size']
    #assert len(embed) + 1 == config['vocab_size'] 
    embed = convert_embed_2_numpy(embed, config['vocab_size'])
    config['embed'] = embed
    #train_rel_file = base_dir + config['relation_train']
    #valid_rel_file = base_dir + config['relation_valid']
    #test_rel_file = base_dir + config['relation_test']

    word_dict = {}
    print query_file
    queries, _ = read_data(query_file)
    print 'Total queries : %d ...'%(len(queries))
    docs, _ =  read_data(doc_file)
    print 'Total docs : %d ...'%(len(docs))

    pair_gen = PairGenerator(data1=queries, data2=docs, config=config)
    list_gen = ListGenerator(data1=queries, data2=docs, config=config)
    #x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls = list_gen.get_all_data()

    model_old = arc_i(config)
    import json
    #json_obj = json.loads(model.to_json())
    json_obj = model_old.get_config()
    json.dump(json_obj, file("arc_i.json", "w"), indent=2)
    model = Model.from_config(json_obj)

    
    #exit(0)

    #for (x1, x1_len, x2, x2_len, y_true) in list_gen.get_batch:
    #   print(y_true)
    #eval_map = MAP_eval(validation_data = list_gen, rel_threshold=0)
    #eval_map = MAP_eval(x1_ls, x2_ls, y_ls, rel_threshold=0)
    rank_eval = rank_eval(rel_threshold = 0.)

    model.compile(optimizer=Adam(lr=config['learning_rate']), loss=rank_hinge_loss)

    num_batch = 10
    train_data_generator = pair_gen.get_batch_generator()

    for k in range(config['num_epochs']):
        model.fit_generator(
                    train_data_generator,
                    steps_per_epoch = num_batch,
                    epochs = 1,
                    verbose = 1
                ) #callbacks=[eval_map])
        res = dict([[k,0.] for k in metrics])
        num_valid = 0
        for (x1, x1_len, x2, x2_len, y_true, _) in list_gen.get_batch:
            y_pred = model.predict({'query': x1, 'doc': x2})
            curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=metrics)
            for k,v in curr_res.items():
                res[k] += v
            num_valid += 1
        print 'epoch: %d,' %( i_e ), '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()]), ' ...'
        sys.stdout.flush()
        list_gen.reset
    



