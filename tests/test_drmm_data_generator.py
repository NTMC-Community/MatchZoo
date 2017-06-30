# -*- coding=utf8 -*-

import os
import sys

sys.path.append('../matchzoo/losses/')
sys.path.append('../matchzoo/metrics/')
sys.path.append('../matchzoo/utils/')
sys.path.append('../matchzoo/models/')
sys.path.append('../matchzoo/inputs/')

from drmm_data_generator import *
from rank_io import *


if __name__=='__main__':
    embed_file = '/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/embed_wiki-pdc_d50_norm'
    query_file = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/qid_query.txt"
    doc_file = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/docid_doc.txt"

    config = {}
    config['vocab_size'] = 193367 + 1
    config['embed_size'] = 50
    config['text1_maxlen'] = 5
    config['hist_size'] = 60
    config['text2_maxlen'] = 50
    config['fill_word'] = 193367
    config['relation_train'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/relation.train.fold1.txt"
    config['relation_test'] = "/data/textnet/data/LetorMQ2007/textnet-letor-mq2007-r5w/relation.test.fold1.txt"
    embed = read_embedding(embed_file)
    embed = convert_embed_2_numpy(embed, config['vocab_size'])

    word_dict = {}
    print query_file
    queries, _ = read_data(query_file)
    print 'Total queries : %d ...'%(len(queries))
    docs, _ =  read_data(doc_file)
    print 'Total docs : %d ...'%(len(docs))

    pair_gen = DRMM_PairGenerator( embed = embed, data1=queries, data2=docs, config=config)
    train_genfun = pair_gen.get_batch_generator()
