"""Amharic character normalization, sentence splitting and tokenization."""

import re

NUM = "<num>"

# Ethiopic syllables are laid out in rows of 8 code points: the base consonant followed by
# its vowel orders (ä, u, i, a, e, ə, o, wa). Homophone series map row-to-row by offset.
_SERIES_MERGES = {
    0x1210: 0x1200,  # ሐ series -> ሀ series
    0x1280: 0x1200,  # ኀ series -> ሀ series
    0x1220: 0x1230,  # ሠ series -> ሰ series
    0x12D0: 0x12A0,  # ዐ series -> አ series
    0x1340: 0x1338,  # ፀ series -> ጸ series
}


def _build_char_map():
    table = {}
    for src, dst in _SERIES_MERGES.items():
        # Orders 0..6 exist for every series; order 7 (labialized) only where both rows define it.
        for order in range(8):
            s, d = chr(src + order), chr(dst + order)
            if s.isprintable() and d.isprintable():
                table[ord(s)] = d
    # The 4th order (a) of the ሀ and አ rows is pronounced like the 1st order.
    for ch, target in {"ሃ": "ሀ", "ኣ": "አ", "ዓ": "አ", "ኃ": "ሀ", "ሓ": "ሀ"}.items():
        table[ord(ch)] = target
    return table


CHAR_MAP = _build_char_map()

# Sentence terminators: ። (full stop), ፧ (question mark), ASCII ?/!, and the two-character
# stand-ins for ። : "፡፡" (two word separators, the most common form in GPAC) and "::".
_SENT_SPLIT = re.compile(r"[።፧?!]+|፡\s*፡|::+|\n+")
# ፡ is the traditional word separator; treat it as whitespace.
_WORD_SEP = re.compile(r"[፡\s]+")
# Arabic digits and Ge'ez numerals (U+1369..U+137C), with internal separators like 1,000.5.
_NUMBER = re.compile(r"^[0-9፩-፼]+([.,/][0-9፩-፼]+)*$")
# Everything that is not an Ethiopic letter, digit, or Ge'ez numeral is punctuation to strip.
_ETHIOPIC_LETTER = re.compile(r"[ሀ-ፚᎀ-ᎏⶀ-⷟꬀-꬯]")
_STRIP = re.compile(r"[^ሀ-ፚ፩-፼ᎀ-ᎏⶀ-⷟꬀-꬯0-9A-Za-z.,/]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGITS = re.compile(r"[0-9፩-፼]+([.,/][0-9፩-፼]+)*")
_INNER_PUNCT = re.compile(r"[.,/]")


def normalize_chars(text):
    return text.translate(CHAR_MAP)


def split_sentences(text):
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text)) if s]


def tokenize(sentence, stats=None):
    """Return the word tokens of one sentence.

    Punctuation is stripped, numbers become NUM, and tokens without Ethiopic letters
    (Latin words, stray symbols) are dropped. If `stats` is a dict, the number of
    dropped tokens is added to stats["dropped_tokens"].
    """
    tokens = []
    for raw in _WORD_SEP.split(sentence):
        tok = _STRIP.sub("", raw).strip(".,/")
        if not tok:
            continue
        if _NUMBER.match(tok):
            tokens.append(NUM)
        elif _ETHIOPIC_LETTER.search(tok) and not _LATIN.search(tok):
            # Ordinals like 2ኛ keep their suffix: "<num>ኛ".
            tok = _DIGITS.sub(NUM, tok)
            tokens.append(_INNER_PUNCT.sub("", tok))
        elif stats is not None:
            stats["dropped_tokens"] = stats.get("dropped_tokens", 0) + 1
    return _collapse_numbers(_join_abbreviations(tokens))


# Abbreviations whose dots GPAC replaced with spaces ("ዓ.ም" -> "ዓ ም"), in normalized spelling.
# Taken from the most frequent runs of one-syllable tokens in the corpus: ዓ.ም (Ethiopian
# calendar), እ.ኤ.አ (Gregorian calendar), km, kg, m², cm, ወ.ዘ.ተ (etc.), ዶ/ር (Dr.), ፌ.ዲ.ሪ
# (FDRE), ህ.ወ.ሓ.ት (TPLF). Other runs ("ቂ ቂ ቂ", laughter) are not abbreviations and stay split.
ABBREVIATIONS = {"አም", "እኤአ", "እአአ", "ኪሜ", "ኪግ", "ካሜ", "ሴሜ", "ወዘተ", "ዶር", "ፌዲሪ", "ህወሀት"}
_MAX_ABBREV = max(map(len, ABBREVIATIONS))


def _join_abbreviations(tokens):
    """Rejoin known abbreviations split into one-syllable tokens, longest match first, so
    that "ዓ ም እ ኤ አ" becomes ["ዓም", "እኤአ"]. Tokens are matched in normalized spelling
    and joined in their original spelling."""
    out, i = [], 0
    while i < len(tokens):
        for L in range(min(_MAX_ABBREV, len(tokens) - i), 1, -1):
            span = tokens[i:i + L]
            if all(len(t) == 1 for t in span) and normalize_chars("".join(span)) in ABBREVIATIONS:
                out.append("".join(span))
                i += L
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def _collapse_numbers(tokens):
    """Drop a bare NUM that follows a token ending in NUM: the corpus writes 8.4 as "8 4"
    and በ8.4 ("at 8.4") as "በ8 4"."""
    return [t for i, t in enumerate(tokens) if not (t == NUM and i > 0 and tokens[i - 1].endswith(NUM))]


def preprocess(text, normalize=True, stats=None):
    """Turn raw text into a list of token lists, one per sentence."""
    if normalize:
        text = normalize_chars(text)
    out = []
    for sent in split_sentences(text):
        toks = tokenize(sent, stats)
        if toks:
            out.append(toks)
    return out
