"""Amharic stopwords: a curated list of function words, optionally extended by frequency.

No standard Amharic stopword list ships with NLTK or spaCy. The curated list holds
conjunctions, pronouns, copulas, postpositions and demonstratives that are written as
separate words. Prefixed forms (በ-, የ-, ለ-, ከ- attached to nouns) are content-bearing words
and are not listed. Spelling follows the homophone normalization in normalize.py.
"""

from .normalize import normalize_chars

CURATED = """
እና ና ወይም ግን ደግሞ እንዲሁም ስለዚህ ስለሆነም እንጂ ቢሆንም ሆኖም እንደ እንደሆነ
ነው ናቸው ነበር ነበሩ ነች ናት ነኝ ነህ ነሽ ነን ናችሁ አለ አሉ አለች አለው አላቸው የለም የሉም ሆኖ ሆነ
ይህ ይህን ይህም ይሄ ያ ያን ያም እነዚህ እነዚያ እዚህ እዚያ እንዲህ እንዲያ ይኸው
እኔ እኛ አንተ አንቺ እናንተ እሱ እሷ እርሱ እርሷ እሳቸው እርሳቸው እነሱ እነርሱ
ላይ ውስጥ ጋር ጋራ በኋላ በፊት ወደ ከ ስለ እስከ ድረስ አጠገብ ዘንድ ፊት ስር በላይ በታች
ሁሉ ሁሉም ብቻ በጣም ሌላ ሌሎች አንዳንድ ማንኛውም እያንዳንዱ አንድ ሁለት
ምን ማን የት መቼ ለምን እንዴት ስንት የትኛው
ሲሆን ሲል ሲሉ ብለዋል ብሏል ገልጸዋል ተናግረዋል አስታውቀዋል መሆኑን መሆኑ የሆነ የሆኑ ያለው ያሉ ያለ
እንዲሆን እንዲሁ ወዘተ ማለት ደግሞም ገና አሁን
"""


def curated_stopwords():
    return {normalize_chars(w) for w in CURATED.split()}


def stopwords(word_counts=None, top_n=0, curated=True):
    """Curated stopwords, optionally unioned with the top_n most frequent words."""
    out = curated_stopwords() if curated else set()
    if word_counts is not None and top_n:
        out |= {w for w, _ in word_counts.most_common(top_n) if not w.startswith("<")}
    return out
