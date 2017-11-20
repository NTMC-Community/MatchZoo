
# generate match histogram
python test_preparation_for_ranking.py
python test_triletter_preprocess.py ranking

cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/dssm_ranking.config

# test the model
python main.py --phase predict --model_file models/dssm_ranking.config
