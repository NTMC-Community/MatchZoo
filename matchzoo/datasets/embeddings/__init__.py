from pathlib import Path
from .load_glove_embedding import load_glove_embedding

DATA_ROOT = Path(__file__).parent
EMBED_RANK = DATA_ROOT.joinpath('embed_rank.txt')
EMBED_10 = DATA_ROOT.joinpath('embed_10_word2vec.txt')
EMBED_10_GLOVE = DATA_ROOT.joinpath('embed_10_glove.txt')
