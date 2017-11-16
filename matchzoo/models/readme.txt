{
  "net_name": "ARCI”,   // model name
  "global":{
      "model_type": "PY”, // model_type, which could be PY or JSON. Set PY if you want to use Python to define model or JSON if you want to use Keras/json to define model.
      "weights_file": "./models/arci.weights”, // the output path of learnt model weights
      "num_epochs": 10, // number of epochs, which is also number of iterations
      "num_batch": 10, // number of batches in each epoch
      "optimizer": "adam”, // type of optimizer
      "learning_rate": 0.0001 // learning rate
  },
  "inputs": {
    "share": {
        "text1_corpus": "../data/example/ranking/corpus_preprocessed.txt”, // input text data1, which could be preprocessed query/question/sentences
        "text2_corpus": "../data/example/ranking/corpus_preprocessed.txt”, // input text data2, which could be preprocessed document/answer/sentences
        "use_dpool": true, // whether use dynamic pooling
        "fill_word": 193367, // the number of words in vocabulary, which is vocab_size
        "embed_size": 50, // size of embedding
        "vocab_size": 193368, // size of vocabulary + 1, since we added a PADDING word
        "train_embed": false, // whether fine tune word embeddings
        "text1_maxlen": 20, // max length of text1 (query/question)
        "text2_maxlen": 500 // max length of text2 (document/answer)
    },
    "train": {
        "input_type": "PairGenerator",  // type of batch generator. For pairwise ranking training, we use PairGenerator. For pointwise classification, we use PointGenerator
        "phase": "TRAIN”, // phase label
        "use_iter": false, // set true if we want to generate batches dynamically with 'iter' mode; set false if we want to generate batches statically (use more memory)
        "query_per_iter": 50, // sample how many queries to generate pair_list. see the code in inputs/pair_generator.py for details
        "batch_per_iter": 5, // sample how many batches before re-sampling another set of queries. see the code in inputs/pair_generator.py for details
        "batch_size": 100, // size of a batch
        "relation_file": "../data/example/ranking/relation_train.txt” // relation file for train data
    },
    "valid": {
        "input_type": "ListGenerator",  // type of batch generator. For validation and testing, we feed a list of documents under a query. Thus we use ListGenerator.
        "phase": "EVAL”, // phase label
        "batch_list": 10, // size of a batch
        "relation_file": "../data/example/ranking/relation_valid.txt” // relation file for validation data
    },
    "test": {
        "input_type": "ListGenerator”, // type of batch generator. for validation and testing, we feed a list of documents under a query. Thus we use ListGenerator.
        "phase": "EVAL”, // phase label
        "batch_list": 10, // feed how many documents for prediction per batch
        "relation_file": "../data/example/ranking/relation_test.txt” // relation file for test data
    },
    "predict": {
        "input_type": "ListGenerator",  // "PREDICT" phase is model prediction by loading saved weights file
        "phase": "PREDICT”, // phase label
        "batch_list": 10, // feed how many documents for prediction per batch
        "relation_file": "../data/example/ranking/relation_test.txt” // relation file for test data
    }
  },
  "outputs": {
    "predict": {
      "save_format": "TREC”, // saved as TREC format to use TREC_EVAL to compute more metrics if needed
      "save_path": "predict.test.fold1.txt” // path of saved score file
    }
  },
  "model": {
    "model_path": "./models/“, // path of models
    "model_py": “arci.ARCI” // python_file_name.class_name
  },
  "losses": [ "rank_hinge_loss" ], // loss function
  "metrics": [ "ndcg@3", "ndcg@5", "map” ] // metrics
}



