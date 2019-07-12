from .unit import Unit
from matchzoo.utils.bert_utils import \
    is_whitespace, is_control, run_strip_accents


class BertClean(Unit):
    """Clean unit for raw text."""

    def transform(self, input_: str) -> str:
        """
        Process input data from raw terms to cleaned text.

        :param input_: raw textual input.

        :return cleaned_text: cleaned text.
        """
        output = []
        for char in input_:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        cleaned_text = "".join(output)
        return cleaned_text


class StripAccent(Unit):
    """Process unit for text lower case."""

    def transform(self, input_: list) -> list:
        """
        Strips accents from each token.

        :param input_: list of tokens.

        :return tokens: Accent-stripped list of tokens.
        """

        return [run_strip_accents(token) for token in input_]
