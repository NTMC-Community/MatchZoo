import matchzoo as mz


def test_load_data():
    train_data = mz.datasets.wiki_qa.load_data('train', task='ranking')
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
