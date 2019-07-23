import string

from .unit import Unit


class PuncRemoval(Unit):
    """Process unit for remove punctuations."""

    def transform(self, input_: list) -> list:
        """
        Remove punctuations from list of tokens.

        :param input_: list of toekns.

        :return rv: tokens  without punctuation.
        """
        table = str.maketrans({key: None for key in string.punctuation})
        return [item.translate(table) for item in input_]
