from matchzoo.preprocessors import *
import pytest

def test_preprocessors():
    test_text = "This is an Example sentence to BE ! cleaned with digits 31."
    tu = TokenizeUnit()
    out = tu.transform(test_text)
    assert len(out) == 13
    assert 'an' in out
    lu = LowercaseUnit()
    out = lu.transform(out)
    assert len(out) == 13
    assert 'be' in out
    du = DigitRemovalUnit()
    out = du.transform(out)
    assert len(out) == 12
    assert '31' not in out
    pu = PuncRemovalUnit()
    out = pu.transform(out)
    assert len(out) == 10
    assert '!' not in out
    stop_u = StopRemovalUnit()
    out = stop_u.transform(out)
    assert len(out) == 3
    su_porter = StemmingUnit()
    out_porter = su_porter.transform(out)
    assert 'sentenc' in out_porter
    su_lancaster = StemmingUnit(stemmer='lancaster')
    out_lancaster = su_lancaster.transform(out)
    assert 'cle' in out_lancaster
    su_error = StemmingUnit(stemmer='fake')
    with pytest.raises(ValueError):
        su_error.transform(out)
    leu = LemmatizationUnit()
    out = leu.transform(out_porter)
    assert len(out) == 3
    StatefulProcessorUnit.__abstractmethods__=set()
    stateful_unit = StatefulProcessorUnit()
    assert stateful_unit.state == {}
