import nltk
from matchzoo.utils.bert_utils import is_chinese_char, \
    whitespace_tokenize, run_split_on_punc

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


class ChineseTokenize(Unit):
    """Process unit for text containing Chinese tokens."""

    def transform(self, input_: str) -> str:
        """
        Process input data from raw terms to processed text.

        :param input_: raw textual input.

        :return output: text with at least one blank between adjacent
                        Chinese tokens.
        """
        output = []
        for char in input_:
            cp = ord(char)
            if is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class BasicTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        orig_tokens = whitespace_tokenize(input_)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class WordPieceTokenize(Unit):
    """Process unit for text tokenization."""

    def __init__(self, vocab: dict, max_input_chars_per_word: int = 200):
        """Initialization."""
        self.vocab = vocab
        self.unk_token = '[UNK]'
        self.max_input_chars_per_word = max_input_chars_per_word

    def transform(self, input_: list) -> list:
        """
        Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform
         tokenization using the given vocabulary.

        For example:
        >>> input_list = ["unaffable"]
        >>> vocab = {"un": 0, "##aff": 1, "##able":2}
        >>> wordpiece_unit = WordPieceTokenize(vocab)
        >>> output = wordpiece_unit.transform(input_list)
        >>> golden_output = ["un", "##aff", "##able"]
        >>> assert output == golden_output

        :param input_: token list.

        :return tokens: A list of wordpiece tokens.
        """
        output_tokens = []
        for token in input_:
            chars = list(token)
            token_length = len(chars)
            if token_length > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            unknown_suffix = False
            start = 0
            sub_tokens = []
            while start < token_length:
                end = token_length
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    unknown_suffix = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if unknown_suffix:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
