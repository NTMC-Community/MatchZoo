import nltk

from .unit import Unit


class Lemmatization(Unit):
    """Process unit for token lemmatization."""

    def transform(self, input_: list) -> list:
        """
        Lemmatization a sequence of tokens.

        :param input_: list of tokens to be lemmatized.

        :return tokens: list of lemmatizd tokens.
        """
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos='v') for token in input_]
