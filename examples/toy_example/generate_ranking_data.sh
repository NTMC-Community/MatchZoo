#!/bin/bash

# generate matchzoo data for ranking
python test_preparation_for_ranking.py

# 1. download embedding 
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.50d.txt ../../data/toy_example/ranking/
# 2. map word embedding
python gen_w2v.py ../../data/toy_example/ranking/glove.6B.50d.txt ../../data/toy_example/ranking/word_dict.txt ../../data/toy_example/ranking/embed_glove_d50
python norm_embed.py  ../../data/toy_example/ranking/embed_glove_d50 ../../data/toy_example/ranking/embed_glove_d50_norm

# 3. run to generate histogram for DRMM
python test_histogram_generator.py  'ranking'

# 4. run to generate tri-grams for DSSM or CDSSM
python test_triletter_preprocess.py 'ranking'

# 5. run to generate binsum for aNMM
python test_binsum_generator.py 'ranking'

