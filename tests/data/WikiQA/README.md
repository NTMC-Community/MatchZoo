## This is a description about the generation of WikiQA dataset.

You can directly run `sh get_data.sh` to prepare the data. This script will execute the commands as follows:

Firstly, it will download the original dataset from ‘https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip’.
```
wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
```

Secondly, it unpack the source dataset into directory ‘WikiQACorpus’.
```
unzip WikiQACorpus.zip
```

Then, the `filter_query` will filter out all queries which have no correct answers in developing set and testing set.
```
python filter_query
```

Finally, the `` will transform the original dataset into matchzoo format.
```
python transfer_to_mz_format.py
```


