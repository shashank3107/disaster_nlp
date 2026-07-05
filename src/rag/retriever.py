"""
Hybrid retriever: structured graph query + semantic search + graph expansion.

Given a question we run three complementary retrievers and fuse them:

  1. Structured — intersect the KG's inverted indexes for the typed filters in the
     QuerySpec (event ∧ category ∧ severity ∧ location ∧ org). This is the exact,
     graph-native retrieval that pure vector RAG cannot do.

  2. Semantic — FAISS top-k over tweet embeddings for fuzzy, vocabulary-free recall.

  3. Graph expansion — from the top seed tweets, pull k-hop neighbours that share an
     entity/event/category, surfacing related context the question didn't name.

Results are merged with Reciprocal Rank Fusion (RRF), a robust rank-combination
that needs no score calibration across the three very different signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .kg_store import KGStore, TweetDoc
from .embed import VectorIndex
from .query_understanding import QuerySpec

logger = logging.getLogger(__name__)


@dataclass
class Retrieved:
    doc:    TweetDoc
    score:  float
    how:    str          # which retriever(s) surfaced it, for transparency


def _rrf(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, tid in enumerate(ranking):
            fused[tid] = fused.get(tid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, store: KGStore, index: VectorIndex) -> None:
        self.store = store
        self.index = index

    # ── individual retrievers ─────────────────────────────────

    def structured(self, spec: QuerySpec, limit: int = 200) -> List[str]:
        """Intersect inverted indexes for the typed filters (AND across types)."""
        sets = []
        if spec.events:
            sets.append(set().union(*(self.store.by_event.get(e, set()) for e in spec.events)))
        if spec.categories:
            sets.append(set().union(*(self.store.by_category.get(c, set()) for c in spec.categories)))
        if spec.severities:
            sets.append(set().union(*(self.store.by_severity.get(s, set()) for s in spec.severities)))
        if spec.locations:
            sets.append(set().union(*(self.store.by_location.get(l, set()) for l in spec.locations)))
        if spec.organizations:
            sets.append(set().union(*(self.store.by_org.get(o, set()) for o in spec.organizations)))
        if not sets:
            return []
        result = set.intersection(*sets) if len(sets) > 1 else sets[0]
        # Fallback: if the strict AND is empty, relax to the union (better recall).
        if not result and len(sets) > 1:
            result = set().union(*sets)
            logger.info("Structured AND empty; relaxed to OR (%d tweets)", len(result))
        return list(result)[:limit]

    def semantic(self, question: str, k: int = 20) -> List[str]:
        return [tid for tid, _ in self.index.search(question, k=k)]

    def expand(self, seeds: List[str], hops: int = 1, per_seed: int = 5) -> List[str]:
        out: List[str] = []
        for tid in seeds:
            nbrs = self.store.neighbours_tweets(tid, hops=hops)
            out.extend(list(nbrs)[:per_seed])
        return out

    # ── fused retrieval ───────────────────────────────────────

    def retrieve(
        self,
        question: str,
        spec: QuerySpec,
        top_k: int = 8,
        sem_k: int = 20,
        expand_hops: int = 1,
        use_expansion: bool = True,
    ) -> List[Retrieved]:
        struct = self.structured(spec)
        sem    = self.semantic(question, k=sem_k)

        # Graph expansion is seeded by the strongest evidence we have so far.
        seeds = (struct[:5] or sem[:5])
        expand = self.expand(seeds, hops=expand_hops, per_seed=4) if use_expansion else []

        # When the question carries hard filters, intersect semantic hits with the
        # structured set so we stay on-topic; otherwise use semantic recall as-is.
        if struct:
            struct_set = set(struct)
            sem_in = [t for t in sem if t in struct_set]
            sem_ranking = sem_in or sem            # keep some recall if no overlap
        else:
            sem_ranking = sem

        fused = _rrf([struct, sem_ranking, expand])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        how_map = self._provenance(struct, sem, expand)
        out: List[Retrieved] = []
        for tid, score in ranked[:top_k]:
            doc = self.store.get(tid)
            if doc is None:
                continue
            out.append(Retrieved(doc=doc, score=round(score, 5), how=how_map.get(tid, "")))
        logger.info("Retrieved %d tweets (structured=%d, semantic=%d, expanded=%d)",
                    len(out), len(struct), len(sem), len(expand))
        return out

    @staticmethod
    def _provenance(struct, sem, expand) -> Dict[str, str]:
        s_set, m_set, e_set = set(struct), set(sem), set(expand)
        prov: Dict[str, str] = {}
        for tid in s_set | m_set | e_set:
            tags = []
            if tid in s_set: tags.append("structured")
            if tid in m_set: tags.append("semantic")
            if tid in e_set: tags.append("expanded")
            prov[tid] = "+".join(tags)
        return prov
