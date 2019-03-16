import pytest

import matchzoo as mz


@pytest.mark.slow
def test_load_data():
    train_data = mz.datasets.wiki_qa.load_data('train', task='ranking')
    assert len(train_data) == 20360
    train_data, _ = mz.datasets.wiki_qa.load_data('train',
                                                  task='classification')
    assert len(train_data) == 20360

    dev_data = mz.datasets.wiki_qa.load_data('dev', task='ranking',
                                             filter=False)
    assert len(dev_data) == 2733
    dev_data, tag = mz.datasets.wiki_qa.load_data('dev', task='classification',
                                                  filter=True)
    assert len(dev_data) == 1126
    assert tag == [False, True]

    test_data = mz.datasets.wiki_qa.load_data('test', task='ranking',
                                              filter=False)
    assert len(test_data) == 6165
    test_data, tag = mz.datasets.wiki_qa.load_data('test',
                                                   task='classification',
                                                   filter=True)
    assert len(test_data) == 2341
    assert tag == [False, True]


@pytest.mark.slow
def test_load_snli():
    train_data, classes = mz.datasets.snli.load_data('train', 'classification')
    assert len(train_data) == 550152
    x, y = train_data.unpack()
    assert len(x['text_left']) == 550152
    assert len(x['text_right']) == 550152
    assert y.shape == (550152, 4)
    assert classes == ['entailment', 'contradiction', 'neutral', '-']
    dev_data, classes = mz.datasets.snli.load_data('dev', 'classification')
    assert len(dev_data) == 10000
    assert classes == ['entailment', 'contradiction', 'neutral', '-']
    test_data, classes = mz.datasets.snli.load_data('test', 'classification')
    assert len(test_data) == 10000
    assert classes == ['entailment', 'contradiction', 'neutral', '-']

    train_data = mz.datasets.snli.load_data('train', 'ranking')
    x, y = train_data.unpack()
    assert len(x['text_left']) == 550152
    assert len(x['text_right']) == 550152
    assert y.shape == (550152, 1)


def test_load_quora_qp():
    train_data, classes = mz.datasets.quora_qp.load_data(
        'matchzoo/datasets/toy/qp_train_demo.csv',
        'train', 'classification')
    assert len(train_data) == 1000
    x, y = train_data.unpack()
    assert len(x['text_left']) == 1000
    assert len(x['text_right']) == 1000
    assert y.shape == (1000, 2)
    test_data, classes = mz.datasets.quora_qp.load_data(
        'matchzoo/datasets/toy/qp_test_demo.csv',
        'test', 'classification')
    assert len(test_data) == 1000
    assert classes == [False, True]

    train_data = mz.datasets.quora_qp.load_data(
        'matchzoo/datasets/toy/qp_train_demo.csv',
        'train', 'ranking')
    x, y = train_data.unpack()
    assert len(x['text_left']) == 1000
    assert y.shape == (1000, 1)
