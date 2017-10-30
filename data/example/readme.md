# The Input Data Format of MatchZoo Toolkit

## Ranking
+ **corpus.txt**: Each line is corresponding to a document. The first column is document ID. Then the following words are from this document after tokenization.

+ **corpus_preprocessed.txt**: Each line is corresponding to a document. The first column is document id, followed by the ids of words.

+ **relation_train.txt/relation_valid.txt/relation_test.txt**: Each line is "label query_id doc_id", which could be used for experiments including document retrieval, passage retrieval, answer sentence selection, etc. For each query, the documents are sorted by the labels. These labels can be binary or multi-graded.

+ **sample.txt**: Each line is the raw query and raw document text of a document. The format is "label \t query \t document_txt".

+ **word_dict.txt**: The word dictionary. Each line is the word and word_id.

+ **corpus_preprocessed_dssm.txt/word_dict_dssm.txt**: These files have the same format with "corpus_preprocessed.txt/word_dict.txt". Since DSSM uses tri-letter based dictionary, we created seperated files for the word dictionary for DSSM model.

## Classification
For classification, the format of most files are the same with the case in ranking. The only difference is the format of "relation_train.txt/relation_valid.txt/relation_test.txt". Each line is "label document_id1 document_id2". Take paraphrase identification as an example, we want to predict whether two sentences have the same meaning using deep text matching models.

## Sample Input Data
The sample input data can be found under MatchZoo/data/example. You can transfer the raw sample.txt to these files by MatchZoo/examples/test_preparation_for_classify.py for classification or MatchZoo/examples/test_preparation_for_ranking.py for ranking.
