import nltk

from .unit import Unit


class StopRemoval(Unit):
    """
    Process unit to remove stop words.

    Example:
        >>> unit = StopRemoval()
        >>> unit.transform(['a', 'the', 'test'])
        ['test']
        >>> type(unit.stopwords)
        <class 'list'>
    """

    def __init__(self, lang: str = 'english'):
        """Initialization."""
        self._lang = lang
        self._stop = nltk.corpus.stopwords.words(self._lang)

    def transform(self, input_: list) -> list:
        """
        Remove stopwords from list of tokenized tokens.

        :param input_: list of tokenized tokens.
        :param lang: language code for stopwords.

        :return tokens: list of tokenized tokens without stopwords.
        """
        return [token
                for token
                in input_
                if token not in self._stop]

    @property
    def stopwords(self) -> list:
        """
        Get stopwords based on language.

        :params lang: language code.
        :return: list of stop words.
        """
        return self._stop
