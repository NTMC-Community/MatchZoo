
# generate match histogram
python generate_match_hist.py

cd ../matchzoo

# configure the model file
#cd models

# train the model
python --phase train --model_file models/drmm.config

# test the model
python --phase predict --model_file models/drmm.config
