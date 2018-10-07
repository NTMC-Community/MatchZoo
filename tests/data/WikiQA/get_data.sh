#!/bin/bash
# help, dos2unix file
# download the wiki-qa dataset
wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
unzip WikiQACorpus.zip

# filter queries which have no right or wrong answers
python filter_query.py

# transfer the dataset into matchzoo dataset format
python transfer_to_mz_format.py

echo "Done ..."
