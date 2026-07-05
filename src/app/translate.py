"""
Language detection + translation to English (runs *before* classification).

The fine-tuned RoBERTa/DistilBERT checkpoints are English-only, so tweets written
in Indian languages (or any non-English language) would be classified on text the
models never saw. This module detects such tweets and translates them to English
first, leaving the original text on the record for provenance.

Detection is two-pronged (covers the "native scripts + romanized" case):
  * Unicode-script check — reliable for native scripts (Devanagari, Tamil, Bengali,
    Telugu, …). Any Indic character ⇒ translate.
  * Latin-script text — `langdetect` plus a small Hinglish keyword heuristic catch
    romanized Indian-language tweets ("baadh aa gaya, madad karo"). Short, noisy
    tweets make this best-effort, not perfect.

Translation uses deep-translator's GoogleTranslator (free Google endpoint, no API
key). Everything degrades gracefully: if a dependency is missing or a call fails,
the original text is kept and a warning is logged.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Unicode blocks for major scripts used by Indian languages (+ Urdu/Arabic).
_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari — Hindi, Marathi, Nepali, Konkani
    (0x0980, 0x09FF),  # Bengali / Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi — Punjabi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x0600, 0x06FF),  # Arabic — Urdu
]

# Common romanized-Hindi/Hinglish cues. A couple of hits flag the tweet as Hindi
# so langdetect's weakness on transliterated text doesn't silently skip it.
_HINGLISH_CUES = {
    "baadh", "paani", "madad", "bachao", "bachaao", "log", "logon", "gaya", "gayi",
    "raha", "rahe", "rahi", "nahi", "nahin", "hai", "hain", "kya", "kyun", "kripya",
    "kripaya", "sahayata", "fasey", "fanse", "phase", "doobe", "doob", "ghar",
    "shahar", "gaon", "barish", "barsat", "toofan", "bhukamp", "aag", "raahat",
    "mrityu", "ghayal", "lapata", "zindagi", "jaan", "sadak", "pul",
}
_TOKEN = re.compile(r"[a-zA-Zऀ-෿؀-ۿ']+")


def _has_indic_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _INDIC_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _looks_hinglish(text: str) -> bool:
    toks = {t.lower() for t in _TOKEN.findall(text)}
    return len(toks & _HINGLISH_CUES) >= 2


def _detect_latin_lang(text: str) -> str:
    """Best-effort language code for Latin-script text ('en' if unsure)."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic
        # langdetect is unreliable on very short strings; treat those as English.
        if len(text.strip()) < 12:
            return "en"
        return detect(text)
    except Exception:
        return "en"


def detect_language(text: str) -> str:
    """Return a coarse language code: 'en', a langdetect code, or 'hi' for Hinglish."""
    if not text or not text.strip():
        return "en"
    if _has_indic_script(text):
        return "auto"          # let the translator auto-detect the exact script
    if _looks_hinglish(text):
        return "hi"
    return _detect_latin_lang(text)


def needs_translation(text: str) -> Tuple[bool, str]:
    """(should_translate, detected_lang). English / empty text ⇒ (False, 'en')."""
    lang = detect_language(text)
    return (lang not in ("en", "unknown")), lang


class Translator:
    """Thin wrapper over deep-translator's GoogleTranslator with graceful fallback."""

    def __init__(self, target: str = "en") -> None:
        self.target = target
        self._gt = None
        self._available = None  # tri-state until first checked

    def available(self) -> bool:
        if self._available is None:
            try:
                from deep_translator import GoogleTranslator  # noqa: F401
                self._available = True
            except Exception as e:
                logger.warning("deep-translator not available: %s", e)
                self._available = False
        return self._available

    def _engine(self):
        if self._gt is None:
            from deep_translator import GoogleTranslator
            # source='auto' lets Google detect the actual language per text.
            self._gt = GoogleTranslator(source="auto", target=self.target)
        return self._gt

    def translate(self, text: str) -> str:
        """Translate one string to English; returns the original on any failure."""
        if not text or not text.strip() or not self.available():
            return text
        try:
            out = self._engine().translate(text)
            return out or text
        except Exception as e:
            logger.warning("Translation failed (%s); keeping original.", e)
            return text


def translate_records(
    records: List[Dict],
    translator: Optional[Translator] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, int]:
    """
    Detect non-English tweets and translate them to English *in place*.

    For every record we set:
        text_original — the original tweet text (always)
        lang          — detected language code ('en' if not translated)
        translated    — bool, whether `text` was replaced with a translation
    `text` itself is overwritten with the English translation when applicable, so
    classification, NER and the KG all run on English. Returns a small summary.
    """
    translator = translator or Translator()
    total = len(records)
    summary = {"total": total, "translated": 0, "skipped_english": 0}

    if total and not translator.available():
        logger.warning("Translator unavailable — skipping translation entirely.")

    for i, r in enumerate(records):
        text = str(r.get("text", ""))
        r.setdefault("text_original", text)
        do_tx, lang = needs_translation(text)
        if do_tx and translator.available():
            translated = translator.translate(text)
            if translated and translated != text:
                r["text"] = translated
                r["lang"] = lang
                r["translated"] = True
                summary["translated"] += 1
            else:
                r["lang"] = lang
                r["translated"] = False
        else:
            r["lang"] = "en"
            r["translated"] = False
            summary["skipped_english"] += 1
        if progress_cb and (i % 10 == 0 or i == total - 1):
            progress_cb(f"translating {i + 1}/{total}", (i + 1) / max(total, 1))

    logger.info("Translation: %d/%d translated, %d English/skipped.",
                summary["translated"], total, summary["skipped_english"])
    return summary
