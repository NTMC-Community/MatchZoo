# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preparation import *
from preprocess import *


if __name__ == '__main__':
    basedir = '../../data/toy_example/classification/'

    # transform query/document pairs into corpus file and relation file
    prepare = Preparation()
    corpus, rels = prepare.run_with_one_corpus( basedir + 'sample.txt')
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(basedir + 'corpus.txt', corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.4, 0.3, 0.3])
    prepare.save_relation(basedir + 'relation_train.txt', rel_train)
    prepare.save_relation(basedir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(basedir + 'relation_test.txt', rel_test)
    print('preparation finished ...')

    # Prerpocess corpus file
    preprocessor = Preprocess()

    dids, docs = preprocessor.run(basedir + 'corpus.txt')
    preprocessor.save_word_dict(basedir + 'word_dict.txt')
    preprocessor.save_words_stats(basedir + 'word_stats.txt')

    fout = open(basedir + 'corpus_preprocessed.txt','w')
    for inum,did in enumerate(dids):
        fout.write('%s %d %s\n'%(did, len(docs[inum]), ' '.join(map(str,docs[inum]))))
    fout.close()
    print('preprocess finished ...')

