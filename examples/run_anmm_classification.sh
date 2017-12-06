cd ../matchzoo

# configure the model file
#cd models

# train the model
python main.py --phase train --model_file models/anmm_classify.config

# test the model
# notice here, in the anmm_classify.config, the weights file for test should be the correct file name, you can find the weights file in MatchZoo/matchzoo/models/weights/.
python main.py --phase predict --model_file models/anmm_classify.config
