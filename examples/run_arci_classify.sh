# generate data for pairs of text

python test_preparation_for_classify.py

cd ../matchzoo

# train the model
python main.py --phase train --model_file models/arci_classify.config


# predict with the model

python main.py --phase predict --model_file models/arci_classify.config
