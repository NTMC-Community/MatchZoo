from .unit import Unit


class Lowercase(Unit):
    """Process unit for text lower case."""

    def transform(self, input_: list) -> list:
        """
        Convert list of tokens to lower case.

        :param input_: list of tokens.

        :return tokens: lower-cased list of tokens.
        """
        return [token.lower() for token in input_]
