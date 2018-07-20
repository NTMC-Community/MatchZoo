#!/usr/bin/env bash

currpath=`pwd`

# In addition to sample.txt, you need to prepare the trec_corpus.txt file that contains all the dataset with trec ids
# that will be used fo train/test/valid
# here is an example line of the trec_corpus.txt file:
# 682 adult immigrant english
# "682" is the query ID followed by its words
# an example is provided in the file: trec_corpus.txt

# data preparation
python3 preparation_for_ranking.py ${currpath}

# download the glove vectors
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# generate word embedding
python3 gen_w2v.py glove.6B.50d.txt word_dict.txt embed_glove_d50
python3 norm_embed.py embed_glove_d50 embed_glove_d50_norm

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
python3 histogram_generator.py 30 50 ${currpath} embed_glove_d50_norm
python3 binsum_generator.py 20 ${currpath} 50 embed_glove_d50_norm # the default number of bin is 20

echo "Done."



