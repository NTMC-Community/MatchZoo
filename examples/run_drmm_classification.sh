
# generate matchzoo data for classification
python test_preparation_for_classify.py

# generate histogram data for drmm
# 1. download embedding
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.50d.txt ../data/example/classification/
# 2. map word embedding
python gen_w2v.py ../data/example/classification/glove.6B.50d.txt ../data/example/classification/word_dict.txt ../data/example/classification/embed_wiki_d50
python norm_embed.py  ../data/example/classification/embed_wiki_d50 ../data/example/classification/embed_wiki_d50_norm
# 3. run to generate histogram
python test_histogram_generator.py classification

cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/drmm_classify.config

# test the model
python main.py --phase predict --model_file models/drmm_classify.config
