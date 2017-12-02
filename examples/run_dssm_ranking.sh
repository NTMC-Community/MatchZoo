
# generate match histogram
python test_preparation_for_ranking.py
python test_triletter_preprocess.py ranking

cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/dssm_ranking.config

# test the model
# notice here, int the dssm_ranking.config, the weights file for test should be the correct file name, you can find the weights file in MatchZoo/matchzoo/models/weights/.
python main.py --phase predict --model_file models/dssm_ranking.config
