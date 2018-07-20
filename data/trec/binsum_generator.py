# /bin/python2.7 and /bin/python3.5
import sys
from os.path import join
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preprocess import *
from rank_io import *

if __name__ == '__main__':
    bin_num = int(sys.argv[1])  # 30
    path = sys.argv[2]
    embed_size = int(sys.argv[3])  # 50
    embedfile = sys.argv[4]  # 'embed_glove_d50_norm'
    corpfile = join(path, 'corpus_preprocessed.txt')
    relfiles = [join(path, 'relation_train.txt'), join(path, 'relation_valid.txt'), join(path, 'relation_test.txt')]
    histfiles = [join(path, 'relation.train.binsum-%d.txt' % bin_num), join(path, 'relation.valid.binsum-%d.txt' % bin_num),
                 join(path, 'relation.test.binsum-%d.txt' % bin_num)]

    # note here word embeddings have been normalized to speed up calculation
    embed_dict = read_embedding(filename=embedfile)
    print('after read embedding ...')
    _PAD_ = len(embed_dict) # for word without wordembeeding, assign an random embedding
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [_PAD_+1, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed=embed)

    corp, _ = read_data(corpfile)
    print('after read corpus ....')

    for i in range(len(relfiles)):
        rel = read_relation(relfiles[i])
        fout = open(histfiles[i], 'w')
        inum = 0
        for label, d1, d2 in rel:
            inum += 1
            assert d1 in corp
            assert d2 in corp
            qnum = len(corp[d1])
            d1_embed = embed[corp[d1]]
            d2_embed = embed[corp[d2]]
            curr_binsum = cal_binsum(d1_embed, d2_embed, qnum, bin_num)
            curr_binsum = curr_binsum.tolist()
            fout.write(' '.join(map(str, curr_binsum)))
            fout.write('\n')
            if inum % 1000 == 0:
                # print('inum: %d ....\r' % inum,)
                sys.stdout.flush()
            # print(curr_hist)
        fout.close()
        print('file: %s processed... '%(relfiles[i]))
    print('\nfinished ...')
