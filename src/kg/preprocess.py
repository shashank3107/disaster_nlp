"""
NER-friendly tweet preprocessing.

This is DELIBERATELY different from the training-time `src.data.preprocess_tweet`:

  * Training cleaning lowercases hashtags and anonymises mentions — great for
    classification, but it destroys exactly the signals NER needs (proper-noun
    casing, hashtag place names like #Houston, organisation handles).

  * Here we keep casing and keep hashtag/mention *words*, only stripping URLs and
    decoding HTML, so that the BERT-NER model sees natural-looking text.

We also expose helpers to pull hashtags and @mentions out separately, since those
become their own nodes in the graph.
"""

from __future__ import annotations

import html
import re
from typing import List, Tuple

_URL_RE     = re.compile(r"http\S+|www\.\S+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MENTION_RE = re.compile(r"@(\w+)")
_WS_RE      = re.compile(r"\s+")


def clean_for_ner(text: str) -> str:
    """
    Light cleaning that preserves NER signal.

    Removes URLs and HTML entities, drops the leading '#'/'@' symbols (so the
    bare word is fed to the tagger) but keeps the word and its original casing.
    """
    t = str(text)
    t = html.unescape(t)
    t = _URL_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(r"\1", t)      # "#Houston" -> "Houston"
    t = _MENTION_RE.sub(r"\1", t)      # "@RedCross" -> "RedCross"
    t = _WS_RE.sub(" ", t)
    return t.strip()


def extract_hashtags(text: str) -> List[str]:
    """Return the list of hashtag words (without '#'), original casing kept."""
    return _HASHTAG_RE.findall(str(text))


def extract_mentions(text: str) -> List[str]:
    """Return the list of @-mention handles (without '@')."""
    return _MENTION_RE.findall(str(text))


def split_camel(token: str) -> str:
    """
    Turn a CamelCase hashtag into spaced words: 'PrayForMexico' -> 'Pray For Mexico'.
    Useful when surfacing hashtags as readable node labels.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)
