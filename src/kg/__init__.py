"""
Knowledge-Graph construction pipeline for CrisisMMD disaster tweets.

This package implements *Approach 1* from the disaster-KG literature:
the classic NLP pipeline

    classified tweets
        -> Named Entity Recognition  (BERT-NER)
        -> Relation Extraction        (schema / rule based)
        -> Subject-Predicate-Object triples
        -> Graph construction + entity alignment  (networkx)
        -> Export (GraphML / JSON / Neo4j Cypher) + visualisation

Each stage is a standalone module so it can be unit-tested or swapped
(e.g. replace the BERT-NER component with a fine-tuned BERT-BiLSTM-CRF),
and `pipeline_kg.py` chains them together end-to-end.
"""

from .schema import (
    EntityType,
    Relation,
    Triple,
    Entity,
    NODE_STYLE,
    HUMANITARIAN_RESPONSE_CATEGORIES,
)

__all__ = [
    "EntityType",
    "Relation",
    "Triple",
    "Entity",
    "NODE_STYLE",
    "HUMANITARIAN_RESPONSE_CATEGORIES",
]
