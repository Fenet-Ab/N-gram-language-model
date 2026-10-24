"""Word clouds for Amharic n-grams.

The default WordCloud font has no Ethiopic glyphs and renders every word as empty boxes,
so an Ethiopic font is located on the system or downloaded (Colab has none installed).
"""

import glob
import os
import urllib.request

FONT_URL = ("https://github.com/google/fonts/raw/main/ofl/notosansethiopic/"
            "NotoSansEthiopic%5Bwdth,wght%5D.ttf")
_SEARCH = ["/usr/share/fonts/**/*Ethiopic*.ttf", "/usr/share/fonts/**/Abyssinica*.ttf",
           os.path.expanduser("~/.fonts/**/*Ethiopic*.ttf")]


def ethiopic_font(cache_dir="."):
    for pattern in _SEARCH:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    path = os.path.join(cache_dir, "NotoSansEthiopic.ttf")
    if not os.path.exists(path):
        urllib.request.urlretrieve(FONT_URL, path)
    return path


def ngram_wordcloud(freqs, ax=None, title=None, max_words=100, font_path=None):
    """Draw a word cloud from {phrase: count}; phrases may contain spaces."""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    wc = WordCloud(font_path=font_path or ethiopic_font(), width=800, height=400,
                   background_color="white", max_words=max_words, collocations=False)
    wc.generate_from_frequencies(freqs)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
