from matchzoo.preprocessors import *


def test_preprocessors():
    processor = chain(tokenizer,
                  to_lowercase,
                  remove_punctuation,
                  remove_digits,
                  stemming,
                  lemmatization)
    rv = processor("This is an Example sentence to BE ! cleaned with digits 31.")
    assert len(rv) == 10
    stoplist = get_stopwords('zh')
    assert len(stoplist) == 119
    terms = remove_stopwords(rv, lang='en')
    assert len(terms) == 5
    sent = '这是一个测试句子．'
    segmented = segmentation(sent)
    assert len(segmented) == 5
