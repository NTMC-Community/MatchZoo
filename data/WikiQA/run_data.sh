#!/bin/bash
# help, dos2unix file
# download the wiki-qa dataset
wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
unzip WikiQACorpus.zip

# download the glove vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# filter queries which have no right or wrong answers
python filter_query.py

# transfer the dataset into matchzoo dataset format
python transfer_to_mz_format.py
# generate the mz-datasets
python prepare_mz_data.py

# generate word embedding
python gen_w2v.py glove.840B.300d.txt word_dict.txt embed_glove_d300
python norm_embed.py embed_glove_d300 embed_glove_d300_norm
python gen_w2v.py glove.6B.50d.txt word_dict.txt embed_glove_d50
python norm_embed.py embed_glove_d50 embed_glove_d50_norm

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
python gen_hist4drmm.py 60
python gen_binsum4anmm.py 20 # the default number of bin is 20

echo "Done ..."
