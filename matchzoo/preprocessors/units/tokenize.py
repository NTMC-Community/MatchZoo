import nltk

from .unit import Unit


class Tokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return nltk.word_tokenize(input_)
