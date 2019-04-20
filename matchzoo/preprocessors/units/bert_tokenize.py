
from .unit import Unit
from .bert_tokenization import FullTokenizer

class BertTokenize(Unit):
    """Process unit for text tokenization by bert tokenizer.

        :param vocab_path: vocab file path in bert pretrained model.
        """

    def __init__(self, vocab_path, do_lower_case=False):
        """Initialization."""
        self.tokenizer = FullTokenizer(vocab_path, do_lower_case=do_lower_case)

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return self.tokenizer.tokenize(input_)
