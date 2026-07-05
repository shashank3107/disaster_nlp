"""
Query understanding: turn a free-text question into typed graph filters.

This is what lets the system exploit the KG's structure instead of treating the
question as a bag of words. We extract:

  * event     — matched against the KG's Event vocabulary (+ common aliases)
  * category  — humanitarian category, via keyword cues
  * severity  — damage severity, via keyword cues
  * locations / organizations — surface forms matched against the KG vocabulary,
    backed up by the same BERT-NER used to build the graph.

Everything is grounded in the *actual* vocabulary present in the loaded KG
(`KGStore.vocab()`), so we never invent a filter that can't match a node.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .kg_store import KGStore

logger = logging.getLogger(__name__)

# Keyword cues -> humanitarian category label (the KG's 8-class vocabulary).
_CATEGORY_CUES = {
    "rescue_volunteering_or_donation_effort":
        ["rescue", "volunteer", "donat", "relief", "aid", "fund", "help", "supplies", "shelter"],
    "infrastructure_and_utility_damage":
        ["infrastructure", "utility", "power", "road", "bridge", "building", "damage to"],
    "injured_or_dead_people":
        ["injured", "dead", "death", "killed", "casualt", "fatalit", "wounded"],
    "affected_individuals":
        ["affected", "victim", "displaced", "evacuat", "stranded"],
    "missing_or_found_people":
        ["missing", "found", "trapped", "search for"],
    "vehicle_damage":
        ["vehicle", "car ", "truck", "boat", "plane"],
    "other_relevant_information":
        ["warning", "update", "forecast", "advisory"],
}

# Keyword cues -> damage severity label.
_SEVERITY_CUES = {
    "severe_damage": ["severe", "destroyed", "devastat", "catastroph", "collapse", "razed"],
    "mild_damage":   ["mild", "minor", "moderate", "partial"],
    "little_or_no_damage": ["little damage", "no damage", "intact", "undamaged"],
}

# A few friendly aliases mapping natural phrasing to the KG's event labels.
_EVENT_ALIASES = {
    "harvey": "hurricane harvey", "irma": "hurricane irma", "maria": "hurricane maria",
    "mexico": "mexico earthquake", "mexico city": "mexico earthquake",
    "iran": "iraq iran earthquake", "iraq": "iraq iran earthquake",
    "sri lanka": "srilanka floods", "srilanka": "srilanka floods",
    "california": "california wildfires", "wildfire": "california wildfires",
    "wildfires": "california wildfires", "puerto rico": "hurricane maria",
}


@dataclass
class QuerySpec:
    """Structured filters extracted from a question."""
    raw:        str
    events:     List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    severities: List[str] = field(default_factory=list)
    locations:  List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)

    def has_structured_filters(self) -> bool:
        return any([self.events, self.categories, self.severities,
                    self.locations, self.organizations])

    def describe(self) -> str:
        bits = []
        for name in ("events", "categories", "severities", "locations", "organizations"):
            v = getattr(self, name)
            if v:
                bits.append(f"{name}={v}")
        return "; ".join(bits) if bits else "(no structured filters)"


class QueryUnderstanding:
    """Parses questions into QuerySpecs grounded in the KG vocabulary."""

    def __init__(self, store: KGStore, use_ner: bool = True,
                 ner_model: str = "dslim/bert-base-NER", device: Optional[str] = None) -> None:
        self.vocab = store.vocab()
        self._ner = None
        if use_ner:
            try:
                from ..kg.ner import EntityExtractor
                self._ner = EntityExtractor(model_name=ner_model, device=device,
                                            use_model=True)
            except Exception as e:
                logger.warning("Query NER unavailable (%s); using vocab matching only.", e)

    def parse(self, question: str) -> QuerySpec:
        q = question.strip()
        low = q.lower()
        spec = QuerySpec(raw=q)

        # events: alias hits + direct vocabulary substring matches
        for alias, ev in _EVENT_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", low) and ev in self.vocab["events"]:
                spec.events.append(ev)
        for ev in self.vocab["events"]:
            if ev in low and ev not in spec.events:
                spec.events.append(ev)

        # category / severity via keyword cues
        for cat, cues in _CATEGORY_CUES.items():
            if any(c in low for c in cues):
                spec.categories.append(cat)
        for sev, cues in _SEVERITY_CUES.items():
            if any(c in low for c in cues):
                spec.severities.append(sev)

        # locations / organizations: KG vocab substring + NER backup
        spec.locations     = self._match_vocab(low, self.vocab["locations"])
        spec.organizations = self._match_vocab(low, self.vocab["organizations"])
        self._augment_with_ner(q, spec)

        # de-duplicate
        for name in ("events", "categories", "severities", "locations", "organizations"):
            setattr(spec, name, sorted(set(getattr(spec, name))))
        logger.info("QuerySpec: %s", spec.describe())
        return spec

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _match_vocab(low_q: str, vocab: List[str], min_len: int = 4) -> List[str]:
        hits = []
        for v in vocab:
            if len(v) >= min_len and re.search(rf"\b{re.escape(v)}\b", low_q):
                hits.append(v)
        return hits

    def _augment_with_ner(self, question: str, spec: QuerySpec) -> None:
        if self._ner is None:
            return
        from ..kg.schema import EntityType
        for ent in self._ner.extract(question):
            v = ent.norm
            if ent.type == EntityType.LOCATION and v in self.vocab["locations"]:
                spec.locations.append(v)
            elif ent.type == EntityType.ORGANIZATION and v in self.vocab["organizations"]:
                spec.organizations.append(v)
