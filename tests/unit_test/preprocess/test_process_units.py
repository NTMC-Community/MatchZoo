from matchzoo.preprocessor.process_units import *
import pytest

@pytest.fixture
def raw_input():
    return "This is an Example sentence to BE ! cleaned with digits 31."

@pytest.fixture
def list_input():
    return ['this', 'Is', 'a', 'the', 'test', 'lIst', '36', '!', 'input']

def test_tokenize_unit(raw_input):
    tu = TokenizeUnit()
    out = tu.transform(raw_input)
    assert len(out) == 13
    assert 'an' in out

def test_lowercase_unit(list_input):
    lu = LowercaseUnit()
    out = lu.transform(list_input)
    assert 'is' in out

def test_digitremoval_unit(list_input):
    du = DigitRemovalUnit()
    out = du.transform(list_input)
    assert 36 not in out

def test_puncremoval_unit(list_input):
    pu = PuncRemovalUnit()
    out = pu.transform(list_input)
    assert '!' not in out

def test_stopremoval_unit(list_input):
    su = StopRemovalUnit()
    out = su.transform(list_input)
    assert 'the' not in out

def test_stemming_unit(list_input):
    su_porter = StemmingUnit()
    out_porter = su_porter.transform(list_input)
    assert 'thi' in out_porter
    su_lancaster = StemmingUnit(stemmer='lancaster')
    out_lancaster = su_lancaster.transform(list_input)
    assert 'thi' in out_lancaster
    su_not_exist = StemmingUnit(stemmer='fake_stemmer')
    with pytest.raises(ValueError):
        su_not_exist.transform(list_input)

def test_lemma_unit(list_input):
    lemma = LemmatizationUnit()
    out = lemma.transform(list_input)
    assert 'this' in out

def test_ngram_unit(list_input):
    ngram = NgramLetterUnit()
    ngram.fit(list_input)
    out = ngram.transform(list_input)
    context = ngram.state
    assert '#a#' in out
    assert context.get('input_dim') == 24
