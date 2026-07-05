"""
Stage 3 — Relation Extraction → Triples.

For tweet-level disaster data, open relation extraction is unreliable (tweets are
short and ungrammatical). Following the spatio-temporal disaster-KG literature, we
use *schema-based* relation extraction: the tweet classification result and the
extracted entities are slotted into a fixed predicate vocabulary (see schema.py).

This yields, per tweet, a set of Subject-Predicate-Object triples with provenance:

    (Tweet)        -[REPORTS]->          (Event)
    (Tweet)        -[HAS_CATEGORY]->     (HumanitarianCategory)
    (Tweet)        -[HAS_SEVERITY]->     (DamageSeverity)
    (Tweet)        -[MENTIONS_LOCATION]->(Location)
    (Tweet)        -[MENTIONS_ORG]->     (Organization)
    (Tweet)        -[MENTIONS_PERSON]->  (Person)
    (Tweet)        -[TAGGED]->           (Hashtag)

Aggregated/inferred edges (Event->Location, Event->Category, Org->Event) are added
later in graph_builder, once all tweets have been seen.
"""

from __future__ import annotations

from typing import Dict, List

from .schema import EntityType, Relation, Triple, Entity, normalise_key

# Labels that are not worth materialising as nodes (no information value).
_SKIP_CATEGORY = {"not_humanitarian", "", "dont_know_or_cant_judge"}
_SKIP_DAMAGE   = {"", "dont_know_or_cant_judge"}

_NER_PRED = {
    EntityType.LOCATION:     Relation.MENTIONS_LOCATION,
    EntityType.ORGANIZATION: Relation.MENTIONS_ORG,
    EntityType.PERSON:       Relation.MENTIONS_PERSON,
    EntityType.HASHTAG:      Relation.TAGGED,
}


def tweet_node_id(tweet_id: str) -> str:
    return f"{EntityType.TWEET.value}:{tweet_id}"


def event_node_id(event: str) -> str:
    return f"{EntityType.EVENT.value}:{normalise_key(event)}"


def extract_triples(record: Dict, entities: List[Entity]) -> List[Triple]:
    """
    Build the triples for a single classified tweet.

    `record`   : output of the classify stage (tweet_id, event, humanitarian, damage…)
    `entities` : output of the NER stage for this tweet
    """
    tid   = str(record.get("tweet_id", ""))
    event = str(record.get("event", "") or "unknown_event")
    subj  = tweet_node_id(tid)
    triples: List[Triple] = []

    # (Tweet) -[REPORTS]-> (Event)
    triples.append(Triple(
        subj=subj, subj_type=EntityType.TWEET,
        pred=Relation.REPORTS,
        obj=event_node_id(event), obj_type=EntityType.EVENT,
        tweet_id=tid, event=event,
    ))

    # (Tweet) -[HAS_CATEGORY]-> (HumanitarianCategory)
    cat = str(record.get("humanitarian", "")).strip()
    if cat not in _SKIP_CATEGORY:
        triples.append(Triple(
            subj=subj, subj_type=EntityType.TWEET,
            pred=Relation.HAS_CATEGORY,
            obj=f"{EntityType.HUMANITARIAN_CATEGORY.value}:{cat}",
            obj_type=EntityType.HUMANITARIAN_CATEGORY,
            tweet_id=tid, event=event,
        ))

    # (Tweet) -[HAS_SEVERITY]-> (DamageSeverity)
    dmg = str(record.get("damage", "")).strip()
    if dmg not in _SKIP_DAMAGE:
        triples.append(Triple(
            subj=subj, subj_type=EntityType.TWEET,
            pred=Relation.HAS_SEVERITY,
            obj=f"{EntityType.DAMAGE_SEVERITY.value}:{dmg}",
            obj_type=EntityType.DAMAGE_SEVERITY,
            tweet_id=tid, event=event,
        ))

    # (Tweet) -[MENTIONS_*]-> (Entity)
    for ent in entities:
        pred = _NER_PRED.get(ent.type)
        if pred is None:
            continue
        triples.append(Triple(
            subj=subj, subj_type=EntityType.TWEET,
            pred=pred,
            obj=ent.node_id, obj_type=ent.type,
            tweet_id=tid, event=event, confidence=ent.score,
        ))

    return triples
