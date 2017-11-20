
# generate matchzoo data for ranking
python test_preparation_for_ranking.py

# generate histogram data for drmm
# 1. download embedding 
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.50d.txt ../data/example/ranking/
# 2. map word embedding
python gen_w2v.py ../data/example/ranking/glove.6B.50d.txt ../data/example/ranking/word_dict.txt ../data/example/ranking/embed_wiki_d50
python norm_embed.py  ../data/example/ranking/embed_wiki_d50 ../data/example/ranking/embed_wiki_d50_norm
# 3. run to generate histogram
python test_histogram_generator.py 

cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/drmm_ranking.config

# test the model
python main.py --phase predict --model_file models/drmm_ranking.config
