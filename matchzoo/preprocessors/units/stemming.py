import nltk

from .unit import Unit


class Stemming(Unit):
    """
    Process unit for token stemming.

    :param stemmer: stemmer to use, `porter` or `lancaster`.
    """

    def __init__(self, stemmer='porter'):
        """Initialization."""
        self.stemmer = stemmer

    def transform(self, input_: list) -> list:
        """
        Reducing inflected words to their word stem, base or root form.

        :param input_: list of string to be stemmed.
        """
        if self.stemmer == 'porter':
            porter_stemmer = nltk.stem.PorterStemmer()
            return [porter_stemmer.stem(token) for token in input_]
        elif self.stemmer == 'lancaster' or self.stemmer == 'krovetz':
            lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
            return [lancaster_stemmer.stem(token) for token in input_]
        else:
            raise ValueError(
                'Not supported supported stemmer type: {}'.format(
                    self.stemmer))
