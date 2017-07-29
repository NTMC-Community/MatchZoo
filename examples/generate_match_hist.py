# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('./matchzoo/inputs')
sys.path.append('./matchzoo/utils')
from rank_io import *
from preprocess import *


# generate histogram of drmm
# cd matchzoo/inputs

def generate_hist():
    embedfile = '../data/mq2007/embed_wiki-pdc_d50_norm'
    queryfile = '../data/mq2007/qid_query.txt'
    docfile = '../data/mq2007/docid_doc.txt'
    relfile = '../data/mq2007/relation.test.fold1.txt'
    histfile = '../data/mq2007/relation.test.fold1.hist-30.txt'
    embed_dict = read_embedding(filename = embedfile)
    print('after read embedding ...')
    _PAD_ = 193367
    embed_dict[_PAD_] = np.zeros((50, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [193368, 50]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    query, _ = read_data(queryfile)
    print('after read query ....')
    doc, _ = read_data(docfile)
    print('after read doc ...')
    rel = read_relation(relfile)
    print('after read relation ... ')
    fout = open(histfile, 'w')
    for label, d1, d2 in rel:
        assert d1 in query
        assert d2 in doc
        qnum = len(query[d1])
        d1_embed = embed[query[d1]]
        d2_embed = embed[doc[d2]]
        curr_hist = cal_hist(d1_embed, d2_embed, qnum, 30)
        curr_hist = curr_hist.tolist()
        fout.write(' '.join(map(str, curr_hist)))
        print(qnum)
        #print(curr_hist)
    fout.close()

if __name__ == '__main__':
    generate_hist()

