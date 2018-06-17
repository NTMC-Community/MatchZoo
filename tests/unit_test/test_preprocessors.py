# from matchzoo.preprocessors import *
# import pytest
# import functools

# def test_preprocessors():
#     test_text = "This is an Example sentence to BE ! cleaned with digits 31."
#     processor = chain(tokenizer,
#                       to_lowercase,
#                       remove_punctuation,
#                       remove_digits,
#                       stemming,
#                       lemmatization)
#     rv = processor(test_text)
#     assert len(rv) == 10
#     stoplist = get_stopwords('zh')
#     assert len(stoplist) == 119
#     terms = remove_stopwords(rv, lang='en')
#     assert len(terms) == 5
#     sent = '这是一个测试句子．'
#     segmented = segmentation(sent)
#     assert len(segmented) == 5
#     processor = chain(tokenizer,
#                       to_lowercase,
#                       functools.partial(stemming, stemmer='lancaster'))
#     rv = processor(test_text)
#     assert 'thi' in rv
#     with pytest.raises(ValueError):
#         stemming(test_text, 'fakestemmer')
                
