"""
KG store: load the disaster knowledge graph and expose it for retrieval.

The graph is loaded from the networkx GraphML produced by `pipeline_kg.py`. Each
Tweet node is turned into a `TweetDoc` — the atomic retrieval unit — carrying the
tweet text plus the typed entities it links to (event, humanitarian category,
damage severity, locations, organisations, persons, hashtags). This pre-joined
view is what both the structured and the semantic retrievers operate on, and it is
also what gets linearised into the LLM context.

The raw networkx graph is kept too, so the retriever can do k-hop expansion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from ..kg.schema import EntityType, Relation

logger = logging.getLogger(__name__)

_TWEET = EntityType.TWEET.value


@dataclass
class TweetDoc:
    """A pre-joined, retrieval-ready view of one tweet and its graph neighbours."""
    tweet_id:     str
    text:         str
    event:        str = ""
    category:     str = ""
    severity:     str = ""
    locations:    List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    persons:      List[str] = field(default_factory=list)
    hashtags:     List[str] = field(default_factory=list)

    def embedding_text(self) -> str:
        """The string we embed: tweet text enriched with its typed context."""
        parts = [self.text]
        if self.event:    parts.append(f"event: {self.event}")
        if self.category and self.category != "not_humanitarian":
            parts.append(f"category: {self.category.replace('_', ' ')}")
        if self.locations: parts.append("locations: " + ", ".join(self.locations))
        if self.organizations: parts.append("orgs: " + ", ".join(self.organizations))
        return " | ".join(parts)

    def context_block(self) -> str:
        """Human/LLM-readable provenance block used in the generation prompt."""
        meta = [f"tweet_id={self.tweet_id}", f"event={self.event or 'n/a'}"]
        if self.category:  meta.append(f"category={self.category}")
        if self.severity:  meta.append(f"severity={self.severity}")
        if self.locations: meta.append("locations=" + ",".join(self.locations))
        if self.organizations: meta.append("orgs=" + ",".join(self.organizations))
        return f"[{'; '.join(meta)}]\n{self.text}"


class KGStore:
    """Loads the KG and builds the TweetDoc index over it."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.g = graph
        self.docs: Dict[str, TweetDoc] = {}
        self._build_docs()
        self._index_typed_nodes()

    # ── loaders ───────────────────────────────────────────────

    @classmethod
    def from_graphml(cls, path: str) -> "KGStore":
        # force_multigraph keeps parallel edges; we read the predicate from edge
        # data (not the multigraph key) so this works whatever type is inferred.
        g = nx.read_graphml(path, force_multigraph=True)
        logger.info("Loaded KG from %s (%d nodes, %d edges)",
                    path, g.number_of_nodes(), g.number_of_edges())
        return cls(g)

    # ── construction ──────────────────────────────────────────

    def _node_label(self, node_id: str) -> str:
        return self.g.nodes[node_id].get("label", node_id.split(":", 1)[-1])

    def _build_docs(self) -> None:
        """Walk every Tweet node and collect its typed out-neighbours."""
        type_to_field = {
            EntityType.LOCATION.value:     "locations",
            EntityType.ORGANIZATION.value: "organizations",
            EntityType.PERSON.value:       "persons",
            EntityType.HASHTAG.value:      "hashtags",
        }
        for n, d in self.g.nodes(data=True):
            if d.get("type") != _TWEET:
                continue
            tid = n.split(":", 1)[-1]
            doc = TweetDoc(tweet_id=tid, text=d.get("text", "") or "")
            for _, nbr, ed in self.g.out_edges(n, data=True):
                key   = ed.get("predicate", "")
                ntype = self.g.nodes[nbr].get("type")
                label = self._node_label(nbr)
                if key == Relation.REPORTS.value:
                    doc.event = label
                elif key == Relation.HAS_CATEGORY.value:
                    doc.category = label
                elif key == Relation.HAS_SEVERITY.value:
                    doc.severity = label
                elif ntype in type_to_field:
                    getattr(doc, type_to_field[ntype]).append(label)
            self.docs[tid] = doc
        logger.info("Built %d TweetDocs from the KG", len(self.docs))

    def _index_typed_nodes(self) -> None:
        """Inverted indexes: which tweets link to a given event/category/location…"""
        self.by_event:    Dict[str, set] = defaultdict(set)
        self.by_category: Dict[str, set] = defaultdict(set)
        self.by_severity: Dict[str, set] = defaultdict(set)
        self.by_location: Dict[str, set] = defaultdict(set)
        self.by_org:      Dict[str, set] = defaultdict(set)

        for tid, doc in self.docs.items():
            if doc.event:    self.by_event[doc.event.lower()].add(tid)
            if doc.category: self.by_category[doc.category.lower()].add(tid)
            if doc.severity: self.by_severity[doc.severity.lower()].add(tid)
            for loc in doc.locations: self.by_location[loc.lower()].add(tid)
            for org in doc.organizations: self.by_org[org.lower()].add(tid)

    # ── accessors ─────────────────────────────────────────────

    def all_docs(self) -> List[TweetDoc]:
        return list(self.docs.values())

    def get(self, tweet_id: str) -> Optional[TweetDoc]:
        return self.docs.get(tweet_id)

    def vocab(self) -> Dict[str, List[str]]:
        """Distinct typed values present in the graph (used by query understanding)."""
        return {
            "events":     sorted(self.by_event),
            "categories": sorted(self.by_category),
            "severities": sorted(self.by_severity),
            "locations":  sorted(self.by_location),
            "organizations": sorted(self.by_org),
        }

    def neighbours_tweets(self, tweet_id: str, hops: int = 1) -> set:
        """
        Tweets reachable from `tweet_id` within `hops` on the undirected graph
        (i.e. tweets that share an entity / event / category with it).
        """
        node = f"{_TWEET}:{tweet_id}"
        if not self.g.has_node(node):
            return set()
        und = self.g.to_undirected(as_view=True)
        ego = nx.ego_graph(und, node, radius=hops)
        out = set()
        for m, dd in ego.nodes(data=True):
            if dd.get("type") == _TWEET and m != node:
                out.add(m.split(":", 1)[-1])
        return out
