# /bin/python2.7
import os
import sys
from os.path import join
import numpy as np
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preparation import *
from preprocess import *


if __name__ == '__main__':
    prepare = Preparation()
    # Prerpocess corpus file
    preprocessor = Preprocess(word_seg_config = { 'enable': True, 'lang': 'en' },
        doc_filter_config = { 'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE },
        word_stem_config = { 'enable': False },
        word_lower_config = { 'enable': False },
        word_filter_config = { 'enable': True, 'stop_words': [], 'min_freq': 1, 'max_freq': six.MAXSIZE, 'words_useless': None })

    basedir = sys.argv[1] #'../../data/toy_example/ranking/'
    try:
        fold = sys.argv[2] # if cross validation, give the folder with file containing all queries and desired documents: trec_corpus.txt
    except:
        fold = -1 
    
    if fold != -1:
        train_file = join(join(basedir, fold), "corpus_train.txt")
        valid_file = join(join(basedir, fold), "corpus_valid.txt")
        test_file = join(join(basedir, fold), "corpus_test.txt")
        corpus_file = join(basedir, "trec_corpus.txt")
        corpus, rel_train, rel_valid, rel_test = prepare.run_with_train_valid_test_corpus_trec(train_file, valid_file, test_file, corpus_file)
        print('total corpus : %d ...' % (len(corpus)))
        prepare.save_corpus(join(join(basedir, fold), 'corpus.txt'), corpus)
        prepare.save_relation(join(join(basedir, fold), 'relation_train.txt'), rel_train)
        prepare.save_relation(join(join(basedir, fold),'relation_valid.txt'), rel_valid)
        prepare.save_relation(join(join(basedir, fold), 'relation_test.txt'), rel_test)
        print('preparation finished ...')
        # pre-processing
        dids, docs = preprocessor.run(join(join(basedir,fold), 'corpus.txt'))
        preprocessor.save_word_dict(join(join(basedir, fold), 'word_dict.txt'))
        preprocessor.save_words_stats(join(join(basedir, fold), 'word_stats.txt'))

        fout = open(join(join(basedir, fold), 'corpus_preprocessed.txt'),'w')
        for inum,did in enumerate(dids):
            fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
        fout.close()

    else:
        # transform query/document pairs into corpus file and relation file
        # corpus, rels = prepare.run_with_one_corpus( basedir + 'sample.txt')
        corpus, rels = prepare.run_with_one_corpus_trec(basedir + 'sample.txt', basedir+"trec_corpus.txt")
        prepare.save_corpus(join(basedir, 'corpus.txt'), corpus)
        print('total relations : %d ...' % (len(rels)))
        rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(rels, [0.6, 0.2, 0.2]) # used wile running with one_corpus
        prepare.save_relation(join(basedir, 'relation_train.txt'), rel_train)
        prepare.save_relation(join(basedir,'relation_valid.txt'), rel_valid)
        prepare.save_relation(join(basedir, 'relation_test.txt'), rel_test)
        print('preparation finished ...')
        dids, docs = preprocessor.run(join(basedir, 'corpus.txt'))
        preprocessor.save_word_dict(join(basedir, 'word_dict.txt'))
        preprocessor.save_words_stats(join(basedir, 'word_stats.txt'))

        fout = open(join(basedir, 'corpus_preprocessed.txt'),'w')
        for inum,did in enumerate(dids):
            fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
        fout.close()

    print('preprocess finished ...')





