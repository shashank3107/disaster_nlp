"""
Knowledge-graph schema: the controlled vocabulary of node and edge types.

Design follows disaster-management ontologies (MOAC, SoKNOS, empathi) but is
deliberately lightweight and tailored to what is *recoverable from tweet text*:
the tweet classification result supplies the typed labels (HumanitarianCategory,
DamageSeverity), CrisisMMD metadata supplies the Event, and BERT-NER supplies the
physical entities (Location / Organization / Person).

Using a *fixed* predicate vocabulary (rather than open relation extraction) keeps
the graph consistent and queryable — the same approach used by the Nature 2026
disaster-storyline KG, which constrains predicates to a small controlled set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Node types
# ─────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    EVENT                 = "Event"                 # e.g. hurricane_harvey
    TWEET                 = "Tweet"                 # a single message
    HUMANITARIAN_CATEGORY = "HumanitarianCategory"  # from the 8-class classifier
    DAMAGE_SEVERITY       = "DamageSeverity"        # from the 3-class classifier
    LOCATION              = "Location"              # NER: LOC / GPE
    ORGANIZATION          = "Organization"          # NER: ORG
    PERSON                = "Person"                # NER: PER
    HASHTAG               = "Hashtag"               # #Harvey, #PrayForMexico


# ─────────────────────────────────────────────────────────────
# Edge types (controlled predicate vocabulary)
# ─────────────────────────────────────────────────────────────

class Relation(str, Enum):
    # Tweet-anchored relations (one per extracted fact)
    REPORTS           = "REPORTS"            # Tweet      -> Event
    HAS_CATEGORY      = "HAS_CATEGORY"       # Tweet      -> HumanitarianCategory
    HAS_SEVERITY      = "HAS_SEVERITY"       # Tweet      -> DamageSeverity
    MENTIONS_LOCATION = "MENTIONS_LOCATION"  # Tweet      -> Location
    MENTIONS_ORG      = "MENTIONS_ORG"       # Tweet      -> Organization
    MENTIONS_PERSON   = "MENTIONS_PERSON"    # Tweet      -> Person
    TAGGED            = "TAGGED"             # Tweet      -> Hashtag

    # Aggregated / inferred relations (built during graph assembly)
    OCCURRED_AT       = "OCCURRED_AT"        # Event      -> Location
    HAS_IMPACT        = "HAS_IMPACT"         # Event      -> HumanitarianCategory
    RESPONDS_TO       = "RESPONDS_TO"        # Organization -> Event


# Humanitarian categories that denote an active response (used to infer the
# Organization -[RESPONDS_TO]-> Event edge when an org is named in such a tweet).
HUMANITARIAN_RESPONSE_CATEGORIES = {
    "rescue_volunteering_or_donation_effort",
}


# ─────────────────────────────────────────────────────────────
# Lightweight dataclasses used by the pipeline stages
# ─────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A typed entity mention extracted from a tweet."""
    text:   str            # raw surface form, e.g. "Houston"
    type:   EntityType     # EntityType.LOCATION
    norm:   str = ""       # normalised key for deduplication (lowercased/cleaned)
    score:  float = 1.0    # extractor confidence

    def __post_init__(self) -> None:
        if not self.norm:
            self.norm = normalise_key(self.text)

    @property
    def node_id(self) -> str:
        """Stable, type-prefixed node identifier used in the graph."""
        return f"{self.type.value}:{self.norm}"


@dataclass
class Triple:
    """A subject-predicate-object fact, with provenance back to the tweet."""
    subj:        str            # subject node_id
    subj_type:   EntityType
    pred:        Relation
    obj:         str            # object node_id
    obj_type:    EntityType
    tweet_id:    str = ""       # provenance
    event:       str = ""
    confidence:  float = 1.0

    def as_dict(self) -> Dict:
        return {
            "subject":     self.subj,
            "subject_type": self.subj_type.value,
            "predicate":   self.pred.value,
            "object":      self.obj,
            "object_type": self.obj_type.value,
            "tweet_id":    self.tweet_id,
            "event":       self.event,
            "confidence":  round(self.confidence, 4),
        }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def normalise_key(text: str) -> str:
    """
    Canonicalise a surface form for entity deduplication.

    "Houston, TX" / "houston" / "HOUSTON"  ->  "houston tx" / "houston"
    Keeps it simple: lowercase, strip punctuation edges, collapse whitespace.
    """
    import re
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)     # drop punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Visualisation styling per node type (colour, size). Consumed by visualize_kg.
NODE_STYLE: Dict[str, Dict] = {
    EntityType.EVENT.value:                 {"color": "#d62728", "size": 1400},
    EntityType.TWEET.value:                 {"color": "#7f7f7f", "size": 120},
    EntityType.HUMANITARIAN_CATEGORY.value: {"color": "#1f77b4", "size": 900},
    EntityType.DAMAGE_SEVERITY.value:       {"color": "#ff7f0e", "size": 900},
    EntityType.LOCATION.value:              {"color": "#2ca02c", "size": 500},
    EntityType.ORGANIZATION.value:          {"color": "#9467bd", "size": 500},
    EntityType.PERSON.value:                {"color": "#8c564b", "size": 400},
    EntityType.HASHTAG.value:               {"color": "#e377c2", "size": 300},
}
