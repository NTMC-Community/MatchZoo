import pytest

import matchzoo as mz


@pytest.mark.cron
def test_load_data():
    train_data = mz.datasets.wiki_qa.load_data('train', task='ranking')
    assert len(train_data) == 20360
    train_data, _ = mz.datasets.wiki_qa.load_data('train',
                                                  task='classification',
                                                  return_classes=True)
    assert len(train_data) == 20360

    dev_data = mz.datasets.wiki_qa.load_data('dev', task='ranking',
                                             filtered=False)
    assert len(dev_data) == 2733
    dev_data, tag = mz.datasets.wiki_qa.load_data('dev', task='classification',
                                                  filtered=True,
                                                  return_classes=True)
    assert len(dev_data) == 1126
    assert tag == [False, True]

    test_data = mz.datasets.wiki_qa.load_data('test', task='ranking',
                                              filtered=False)
    assert len(test_data) == 6165
    test_data, tag = mz.datasets.wiki_qa.load_data('test',
                                                   task='classification',
                                                   filtered=True,
                                                   return_classes=True)
    assert len(test_data) == 2341
    assert tag == [False, True]


@pytest.mark.cron
def test_load_snli():
    train_data, classes = mz.datasets.snli.load_data('train',
                                                     'classification',
                                                     return_classes=True)
    num_samples = 550146
    assert len(train_data) == num_samples
    x, y = train_data.unpack()
    assert len(x['text_left']) == num_samples
    assert len(x['text_right']) == num_samples
    assert y.shape == (num_samples, 4)
    assert classes == ['entailment', 'contradiction', 'neutral', '-']
    dev_data, classes = mz.datasets.snli.load_data('dev', 'classification',
                                                   return_classes=True)
    assert len(dev_data) == 10000
    assert classes == ['entailment', 'contradiction', 'neutral', '-']
    test_data, classes = mz.datasets.snli.load_data('test', 'classification',
                                                    return_classes=True)
    assert len(test_data) == 10000
    assert classes == ['entailment', 'contradiction', 'neutral', '-']

    train_data = mz.datasets.snli.load_data('train', 'ranking')
    x, y = train_data.unpack()
    assert len(x['text_left']) == num_samples
    assert len(x['text_right']) == num_samples
    assert y.shape == (num_samples, 1)


@pytest.mark.cron
def test_load_quora_qp():
    train_data = mz.datasets.quora_qp.load_data(task='classification')
    assert len(train_data) == 363177

    dev_data, tag = mz.datasets.quora_qp.load_data(
        'dev',
        task='classification',
        return_classes=True)
    assert tag == [False, True]
    assert len(dev_data) == 40371
    x, y = dev_data.unpack()
    assert len(x['text_left']) == 40371
    assert len(x['text_right']) == 40371
    assert y.shape == (40371, 2)

    test_data = mz.datasets.quora_qp.load_data('test')
    assert len(test_data) == 390965

    dev_data = mz.datasets.quora_qp.load_data('dev', 'ranking')
    x, y = dev_data.unpack()
    assert y.shape == (40371, 1)


@pytest.mark.cron
def test_load_cqa_ql_16():
    # test load question pairs
    train_data = mz.datasets.cqa_ql_16.load_data(task='classification')
    assert len(train_data) == 3998
    dev_data, tag = mz.datasets.cqa_ql_16.load_data(
        'dev',
        task='classification',
        return_classes=True)
    assert tag == ['PerfectMatch', 'Relevant', 'Irrelevant']
    assert len(dev_data) == 500
    x, y = dev_data.unpack()
    assert y.shape == (500, 3)
    test_data = mz.datasets.cqa_ql_16.load_data('test')
    assert len(test_data) == 700

    # test load answer pairs
    train_data = mz.datasets.cqa_ql_16.load_data(match_type='answer')
    assert len(train_data) == 39980
    test_data = mz.datasets.cqa_ql_16.load_data(stage='test', match_type='answer')
    assert len(test_data) == 7000

    # test load external answer pairs
    train_data = mz.datasets.cqa_ql_16.load_data(match_type='external_answer')
    assert len(train_data) == 39980

    # test load rank data
    train_data = mz.datasets.cqa_ql_16.load_data(task='ranking')
    x, y = train_data.unpack()
    assert y.shape == (3998, 1)

    dev_data = mz.datasets.cqa_ql_16.load_data('dev', task='ranking', match_type='answer', target_label='Good')
    x, y = dev_data.unpack()
    assert y.shape == (5000, 1)
