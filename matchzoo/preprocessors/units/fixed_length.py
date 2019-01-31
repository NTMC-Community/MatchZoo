import typing

import numpy as np

from .unit import Unit


class FixedLength(Unit):
    """
    FixedLengthUnit Class.

    Process unit to get the fixed length text.

    Examples:
        >>> from matchzoo.preprocessors.units import FixedLength
        >>> fixedlen = FixedLength(3)
        >>> fixedlen.transform(list(range(1, 6))) == [3, 4, 5]
        True
        >>> fixedlen.transform(list(range(1, 3))) == [0, 1, 2]
        True

    """

    def __init__(
        self,
        text_length: int,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
        truncate_mode: str = 'pre'
    ):
        """
        Class initialization.

        :param text_length: fixed length of the text.
        :param pad_value: if text length is smaller than :attr:`text_length`,
            filling text with :attr:`pad_value`.
        :param pad_mode: String, `pre` or `post`:
            pad either before or after each sequence.
        :param truncate_mode: String, `pre` or `post`:
            remove values from sequences larger than :attr:`text_length`,
            either at the beginning or at the end of the sequences.
        """
        self._text_length = text_length
        self._pad_value = pad_value
        self._pad_mode = pad_mode
        self._truncate_mode = truncate_mode

    def transform(self, input_: list) -> list:
        """
        Transform list of tokenized tokens into the fixed length text.

        :param input_: list of tokenized tokens.

        :return tokens: list of tokenized tokens in fixed length.
        """
        # padding process can not handle empty list as input
        if len(input_) == 0:
            input_ = [self._pad_value]
        np_tokens = np.array(input_)
        fixed_tokens = np.full([self._text_length], self._pad_value,
                               dtype=np_tokens.dtype)

        if self._truncate_mode == 'pre':
            trunc_tokens = input_[-self._text_length:]
        elif self._truncate_mode == 'post':
            trunc_tokens = input_[:self._text_length]
        else:
            raise ValueError('{} is not a vaild '
                             'truncate mode.'.format(self._truncate_mode))

        if self._pad_mode == 'post':
            fixed_tokens[:len(trunc_tokens)] = trunc_tokens
        elif self._pad_mode == 'pre':
            fixed_tokens[-len(trunc_tokens):] = trunc_tokens
        else:
            raise ValueError('{} is not a vaild '
                             'pad mode.'.format(self._pad_mode))

        return fixed_tokens.tolist()
