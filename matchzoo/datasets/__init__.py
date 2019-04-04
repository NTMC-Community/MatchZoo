from . import toy
from . import wiki_qa
from . import embeddings
from . import snli
from . import quora_qp
from pathlib import Path


def list_available():
    return [p.name for p in Path(__file__).parent.iterdir()
            if p.is_dir() and not p.name.startswith('_')]
