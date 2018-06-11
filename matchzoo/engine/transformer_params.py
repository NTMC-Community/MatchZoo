"""Transformer parametrs."""


class TransformerParams(dict):
    """Transformer parametrs."""

    def __init__(self):
        """Create the default parametrs for transformer."""
        super().__init__(
                name=None,
                transformer_class=None,
                analyzer='word',
                tokenizer=None,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=None,
                vocabulary=None,
                ngram_range=(1, 1)
        )
