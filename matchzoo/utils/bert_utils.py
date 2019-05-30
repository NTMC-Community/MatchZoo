import unicodedata


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    return (char == " ") or \
           (char == "\t") or \
           (char == "\n") or \
           (char == "\r") or \
           (unicodedata.category(char) == "Zs")


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ["Cc", "Cf"]:
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    condition = (33 <= cp <= 47) or (58 <= cp <= 64) or \
                (91 <= cp <= 96) or (123 <= cp <= 126)
    cat = unicodedata.category(char)
    if condition or cat.startswith("P"):
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean
    # characters, despite its name. The modern Korean Hangul alphabet is a
    # different block, as is Japanese Hiragana and Katakana. Those alphabets
    # are used to write space-separated words, so they are not treated
    # specially and handled like the all of the other languages.
    return (0x4E00 <= cp <= 0x9FFF) or \
           (0x3400 <= cp <= 0x4DBF) or \
           (0x20000 <= cp <= 0x2A6DF) or \
           (0x2A700 <= cp <= 0x2B73F) or \
           (0x2B740 <= cp <= 0x2B81F) or \
           (0x2B820 <= cp <= 0x2CEAF) or \
           (0xF900 <= cp <= 0xFAFF) or \
           (0x2F800 <= cp <= 0x2FA1F)


def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = [char for char in text if not unicodedata.category(char) == 'Mn']
    return "".join(output)


def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    tokens = text.split()
    return tokens
