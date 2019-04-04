from .unit import Unit


class DigitRemoval(Unit):
    """Process unit to remove digits."""

    def transform(self, input_: list) -> list:
        """
        Remove digits from list of tokens.

        :param input_: list of tokens to be filtered.

        :return tokens: tokens of tokens without digits.
        """
        return [token for token in input_ if not token.isdigit()]
