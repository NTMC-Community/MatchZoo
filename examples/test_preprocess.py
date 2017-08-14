# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
from preprocess import *


if __name__ == '__main__':
    path = '../../dataset/marco/tmp/'

    # train file
    infile_path = path + 'did.train.txt'
    outfile_path = path + 'did.train.processed.txt'
    dictfile_path = path + 'word_dict.txt'
    dffile_path = path + 'word_df.txt'
    preprocessor = Preprocess(min_freq = 5)
    dids, docs = preprocessor.run(infile_path)
    preprocessor.save_word_dict(dictfile_path)
    preprocessor.save_words_df(dffile_path)

    fout = open(outfile_path,'w')
    for inum,did in enumerate(dids):
        fout.write('%s\t%s\n'%(did, ' '.join(map(str,docs[inum]))))
    fout.close()
    print('Train file finished ...')

    # valid file
    infile_path = path + 'did.dev.txt'
    outfile_path = path + 'did.dev.processed.txt'
    #preprocessor.load_word_dict(dictfile_path)
    #preprocessor.load_words_df(dffile_path)
    dids, docs = preprocessor.run(infile_path)

    fout = open(outfile_path,'w')
    for inum,did in enumerate(dids):
        fout.write('%s\t%s\n'%(did, ' '.join(map(str,docs[inum]))))
    fout.close()
    print('Valid file finished ...')

    '''
    # test file
    infile_path = path + 'did.test.txt'
    outfile_path = path + 'did.test.processed.txt'
    preprocessor = Preprocess(min_freq = 5)
    #preprocessor.load_word_dict(dictfile_path)
    #preprocessor.load_words_df(dffile_path)
    dids, docs = preprocessor.run(infile_path)

    fout = open(outfile_path,'w')
    for inum,did in enumerate(dids):
        fout.write('%s\t%s\n'%(did, ' '.join(map(str,docs[inum]))))
    fout.close()
    print('Test file finished ...')
    '''
