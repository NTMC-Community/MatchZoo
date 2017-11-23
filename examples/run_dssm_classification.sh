
# generate match histogram
python test_preparation_for_classification.py
python test_triletter_preprocess.py classification

cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/dssm_classification.config

# test the model
python main.py --phase predict --model_file models/dssm_classification.config
