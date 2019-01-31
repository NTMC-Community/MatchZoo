import re

from .unit import Unit


class PuncRemoval(Unit):
    """Process unit for remove punctuations."""

    _MATCH_PUNC = re.compile(r'[^\w\s]')

    def transform(self, input_: list) -> list:
        """
        Remove punctuations from list of tokens.

        :param input_: list of toekns.

        :return rv: tokens  without punctuation.
        """
        return [token for token in input_ if
                not self._MATCH_PUNC.search(token)]
