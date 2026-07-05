"""
Stage 2 — Named Entity Recognition.

Implements the "BERT-NER" component of Approach 1. We use a transformer token
classifier (default: dslim/bert-base-NER, fine-tuned on CoNLL-2003 → PER/ORG/LOC/
MISC) via the HuggingFace `pipeline`, with sub-word aggregation so multi-token
spans like "Red Cross" come back as one entity.

Robustness: if the model cannot be downloaded/loaded (e.g. offline compute node),
we fall back to a regex/gazetteer extractor so the pipeline still produces a graph.
Hashtags and @mentions are always extracted with regex and added as their own
entity nodes.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from .schema import Entity, EntityType, normalise_key
from .preprocess import clean_for_ner, extract_hashtags, extract_mentions, split_camel

logger = logging.getLogger(__name__)

# Map the NER model's tags to our schema's entity types.
_TAG2TYPE = {
    "PER":  EntityType.PERSON,
    "PERSON": EntityType.PERSON,
    "ORG":  EntityType.ORGANIZATION,
    "LOC":  EntityType.LOCATION,
    "GPE":  EntityType.LOCATION,
    # "MISC" is intentionally dropped — too noisy for a clean disaster KG.
}

# Minimal gazetteer used by the regex fallback (CrisisMMD's seven events).
_FALLBACK_LOCATIONS = {
    "houston", "texas", "florida", "mexico", "mexico city", "puerto rico",
    "california", "iraq", "iran", "sri lanka", "harvey", "irma", "maria",
}

# Disaster / hurricane names the NER model frequently mislabels as PERSON or ORG.
# They are already captured by the Event node, so we drop them as entities to
# avoid polluting the Person/Organization layers.
_EVENT_NAME_STOPLIST = {
    "harvey", "irma", "maria", "nate", "katia",       # 2017 hurricanes
    "hurricane", "earthquake", "wildfire", "wildfires", "flood", "floods",
}


class EntityExtractor:
    """Wraps a transformer NER pipeline (with graceful fallback)."""

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: Optional[str] = None,
        min_score: float = 0.80,
        use_model: bool = True,
    ) -> None:
        self.min_score = min_score
        self._pipe = None
        if use_model:
            self._pipe = self._build_pipe(model_name, device)
        if self._pipe is None:
            logger.warning("NER running in REGEX-FALLBACK mode (no transformer model).")

    def _build_pipe(self, model_name: str, device: Optional[str]):
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
            dev = 0 if (device != "cpu" and torch.cuda.is_available()) else -1
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForTokenClassification.from_pretrained(model_name)
            pipe = pipeline(
                "token-classification",
                model=mdl,
                tokenizer=tok,
                aggregation_strategy="simple",   # merge sub-words into spans
                device=dev,
            )
            logger.info("Loaded NER model '%s' on device %s", model_name, dev)
            return pipe
        except Exception as e:  # offline / missing weights / OOM
            logger.warning("Could not load NER model '%s': %s", model_name, e)
            return None

    # ── public API ────────────────────────────────────────────

    def extract(self, text: str) -> List[Entity]:
        """Extract typed entities from one tweet (NER spans + hashtags + mentions)."""
        cleaned = clean_for_ner(text)
        entities: List[Entity] = []

        if self._pipe is not None:
            entities.extend(self._extract_model(cleaned))
        else:
            entities.extend(self._extract_regex(cleaned))

        entities.extend(self._extract_tags(text))
        return self._dedup(entities)

    def extract_batch(self, texts: List[str], batch_size: int = 64) -> List[List[Entity]]:
        """Vectorised extraction; returns one entity-list per input tweet."""
        cleaned = [clean_for_ner(t) for t in texts]
        out: List[List[Entity]] = [[] for _ in texts]

        if self._pipe is not None:
            try:
                results = self._pipe(cleaned, batch_size=batch_size)
                # pipeline returns a list-of-lists for list input
                for i, spans in enumerate(results):
                    out[i].extend(self._spans_to_entities(spans))
            except Exception as e:
                logger.warning("Batched NER failed (%s); falling back per-item.", e)
                for i, t in enumerate(cleaned):
                    out[i].extend(self._extract_model(t))
        else:
            for i, t in enumerate(cleaned):
                out[i].extend(self._extract_regex(t))

        for i, raw in enumerate(texts):
            out[i].extend(self._extract_tags(raw))
            out[i] = self._dedup(out[i])
        return out

    # ── extractors ────────────────────────────────────────────

    def _extract_model(self, text: str) -> List[Entity]:
        if not text:
            return []
        try:
            spans = self._pipe(text)
        except Exception:
            return []
        return self._spans_to_entities(spans)

    def _spans_to_entities(self, spans) -> List[Entity]:
        ents: List[Entity] = []
        for s in spans:
            tag   = s.get("entity_group") or s.get("entity") or ""
            tag   = tag.split("-")[-1]            # strip B-/I- prefixes if present
            etype = _TAG2TYPE.get(tag)
            score = float(s.get("score", 1.0))
            word  = self._clean_word(s.get("word") or "")
            if etype is None or score < self.min_score:
                continue
            if not self._is_valid_word(word):
                continue
            if word.lower() in _EVENT_NAME_STOPLIST:   # hurricane/quake names → Event, not entity
                continue
            ents.append(Entity(text=word, type=etype, score=round(score, 4)))
        return ents

    @staticmethod
    def _clean_word(word: str) -> str:
        """Strip WordPiece artifacts ('##') and surrounding whitespace."""
        return word.replace("##", "").strip()

    @staticmethod
    def _is_valid_word(word: str) -> bool:
        """Reject unmerged sub-word fragments and too-short tokens."""
        if len(word) < 3:
            return False
        # Pure fragments like "rri", "in", lowercase-only short stubs from a split
        # proper noun — require at least one uppercase letter (proper nouns) or a space.
        if " " not in word and not any(c.isupper() for c in word):
            return False
        return True

    def _extract_regex(self, text: str) -> List[Entity]:
        """Fallback: gazetteer match for locations + capitalised multiword spans."""
        ents: List[Entity] = []
        low = text.lower()
        for loc in _FALLBACK_LOCATIONS:
            if re.search(rf"\b{re.escape(loc)}\b", low):
                ents.append(Entity(text=loc.title(), type=EntityType.LOCATION, score=0.5))
        # Sequences of Capitalised words → candidate proper nouns (ORG/LOC unknown)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
            ents.append(Entity(text=m.group(1), type=EntityType.ORGANIZATION, score=0.4))
        return ents

    def _extract_tags(self, raw_text: str) -> List[Entity]:
        ents: List[Entity] = []
        for h in extract_hashtags(raw_text):
            label = split_camel(h)
            ents.append(Entity(text=label, type=EntityType.HASHTAG, score=1.0))
        # @mentions of recognisable relief orgs become Organization nodes;
        # others are skipped to avoid flooding the graph with random handles.
        for m in extract_mentions(raw_text):
            if any(k in m.lower() for k in ("redcross", "fema", "unicef", "relief",
                                            "rescue", "gov", "police", "fire")):
                ents.append(Entity(text=m, type=EntityType.ORGANIZATION, score=0.6))
        return ents

    @staticmethod
    def _dedup(entities: List[Entity]) -> List[Entity]:
        """Collapse repeated mentions within a single tweet (keep highest score)."""
        best: Dict[str, Entity] = {}
        for e in entities:
            key = e.node_id
            if key not in best or e.score > best[key].score:
                best[key] = e
        return list(best.values())
