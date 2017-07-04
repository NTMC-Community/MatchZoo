# -*- coding=utf8 -*-

import os
import sys

sys.path.append('../matchzoo/losses/')
sys.path.append('../matchzoo/metrics/')
sys.path.append('../matchzoo/utils/')
sys.path.append('../matchzoo/models/')
sys.path.append('../matchzoo/inputs/')

import pair_generator
import list_generator
from rank_io import *


if __name__=='__main__':
    embed_file = '../data/mq2007/embed_wiki-pdc_d50_norm'
    query_file = "../data/mq2007/qid_query.txt"
    doc_file = "../data/mq2007/docid_doc.txt"

    config = {}
    config['vocab_size'] = 193367 + 1
    config['embed_size'] = 50
    config['text1_maxlen'] = 5
    config['batch_size'] = 2
    config['hist_size'] = 10
    config['use_iter'] = False 
    config['query_per_iter'] = True
    config['text2_maxlen'] = 50
    config['fill_word'] = 193367
    config['relation_train'] = "../data/mq2007/relation.train.fold1.txt"
    config['relation_test'] = "../data/mq2007/relation.test.fold1.txt"
    embed = read_embedding(embed_file)
    embed = convert_embed_2_numpy(embed, config['vocab_size'])
    config['embed'] = embed

    word_dict = {}
    print query_file
    queries, _ = read_data(query_file)
    print 'Total queries : %d ...'%(len(queries))
    docs, _ =  read_data(doc_file)
    print 'Total docs : %d ...'%(len(docs))

    pair_gen = DRMM_PairGenerator( data1=queries, data2=docs, config=config)
    train_genfun = pair_gen.get_batch_generator()
    inum = 0
    for dinputs, y in train_genfun:
        print dinputs['query']
        print dinputs['doc']
        inum += 1
        if inum > 10:
            break
