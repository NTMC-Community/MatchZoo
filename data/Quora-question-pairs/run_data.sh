#!/bin/bash
#Attention；You need to  register the Kaggle account to download the dataset

# download the quora train dataset
#wget --keep-session-cookies --save-cookies cookies.txt --post-data "username=username&password=password" "https://www.kaggle.com/account/login?isModal=true&returnUrl=/"
#wget --load-cookies=cookies.txt  "https://www.kaggle.com/c/quora-question-pairs/download/train.csv.zip"
#unzip train.csv.zip
#download the quora test dataset
#wget --load-cookies=cookies.txt "https://www.kaggle.com/c/quora-question-pairs/download/test.csv.zip"
#unzip test.csv.zip

#You can also download and unzip it manually on the official web, and save it to the current directory

# download the glove vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# generate the mz-datasets
python prepare_mz_data.py

# generate word embedding
python ../WikiQA/gen_w2v.py  glove.840B.300d.txt word_dict.txt embed_glove_d300
python ../WikiQA/norm_embed.py embed_glove_d300 embed_glove_d300_norm
python ../WikiQA/gen_w2v.py  glove.6B.50d.txt word_dict.txt embed_glove_d50
python ../WikiQA/norm_embed.py embed_glove_d50 embed_glove_d50_norm

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
#python ../WikiQA/gen_hist4drmm.py 60
#python gen_binsum4anmm.py 20 # the default number of bin is 20

echo "Done ..."
