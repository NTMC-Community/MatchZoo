from tqdm import tqdm

from . import DataPack
from matchzoo import processor_units


def build_vocab(data_pack: DataPack):
    vocab = []
    data_pack.apply_on_text(vocab.extend)
    vocab_unit = processor_units.VocabularyUnit()
    vocab_unit.fit(tqdm(vocab, desc='Fitting vocabulary unit.'))
    return vocab_unit
