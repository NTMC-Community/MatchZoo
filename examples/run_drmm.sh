
# generate match histogram
python test_histogram_generator.py 

cd ../matchzoo

# configure the model file
#cd models

# train the model
python --phase train --model_file models/drmm.config

# test the model
python --phase predict --model_file models/drmm.config
