"""
Stage 4 — Graph assembly, entity alignment, and export.

Consumes the flat list of triples (from every tweet) and assembles a single
networkx MultiDiGraph:

  * Nodes are deduplicated by their type-prefixed `node_id` (entity alignment by
    normalised surface form — the lightweight surface-matching step used in the
    Scientific Reports 2025 MHKFF fusion framework).
  * Tweet->X edges are added directly from the triples.
  * Aggregated / inferred edges are then materialised across the whole graph:
        Event -[OCCURRED_AT]->  Location           (location co-mentioned with event)
        Event -[HAS_IMPACT]->   HumanitarianCategory
        Org   -[RESPONDS_TO]->  Event              (org named in a response tweet)

Exports:
  * <name>.graphml   — for Gephi / Cytoscape / yEd
  * <name>.json      — node-link JSON (for D3 / inspection)
  * <name>.cypher    — Neo4j load script (MERGE statements; no server needed)
  * <name>.stats.json— summary counts for the report / thesis tables
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx

from .schema import (
    EntityType,
    Relation,
    Triple,
    NODE_STYLE,
    HUMANITARIAN_RESPONSE_CATEGORIES,
)

logger = logging.getLogger(__name__)


def _readable_label(node_id: str) -> str:
    """'Location:houston tx' -> 'houston tx' ; 'Event:hurricane harvey' -> 'hurricane harvey'."""
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


class KnowledgeGraph:
    """Builds and exports the disaster knowledge graph from triples."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ── construction ──────────────────────────────────────────

    def add_triples(self, triples: List[Triple], texts: Dict[str, str] | None = None,
                    meta: Dict[str, Dict] | None = None) -> None:
        """
        Add all Tweet-anchored triples, then derive aggregated edges.

        `meta` (optional) maps tweet_id -> {url, username, timestamp} so Tweet nodes
        can carry provenance for citations, without adding new node types.
        """
        texts = texts or {}
        meta  = meta or {}

        for t in triples:
            self._ensure_node(t.subj, t.subj_type, texts, meta)
            self._ensure_node(t.obj,  t.obj_type,  texts, meta)
            self.g.add_edge(t.subj, t.obj, key=t.pred.value,
                            predicate=t.pred.value,
                            tweet_id=t.tweet_id,
                            confidence=t.confidence)

        self._add_inferred_edges()

    def _ensure_node(self, node_id: str, ntype: EntityType, texts: Dict[str, str],
                     meta: Dict[str, Dict] | None = None) -> None:
        if self.g.has_node(node_id):
            return
        attrs = {
            "type":  ntype.value,
            "label": _readable_label(node_id),
            "color": NODE_STYLE.get(ntype.value, {}).get("color", "#cccccc"),
            "size":  NODE_STYLE.get(ntype.value, {}).get("size", 300),
        }
        if ntype == EntityType.TWEET:
            tid = node_id.split(":", 1)[1]
            attrs["text"]  = texts.get(tid, "")
            attrs["label"] = tid
            for k in ("url", "username", "timestamp"):
                v = (meta or {}).get(tid, {}).get(k)
                if v:
                    attrs[k] = v
        self.g.add_node(node_id, **attrs)

    def _add_inferred_edges(self) -> None:
        """Materialise Event-level aggregations and Org responses."""
        # Map each tweet -> its event, category, and the entities it mentions.
        tweet_event:    Dict[str, str] = {}
        tweet_category: Dict[str, str] = {}

        for u, v, k in self.g.edges(keys=True):
            if k == Relation.REPORTS.value:
                tweet_event[u] = v
            elif k == Relation.HAS_CATEGORY.value:
                tweet_category[u] = v

        event_locs   = defaultdict(set)   # event -> {location_id}
        event_impact = defaultdict(set)   # event -> {category_id}
        org_event    = set()              # (org_id, event_id)

        for u, v, k in self.g.edges(keys=True):
            ev = tweet_event.get(u)
            if ev is None:
                continue
            if k == Relation.MENTIONS_LOCATION.value:
                event_locs[ev].add(v)
            elif k == Relation.HAS_CATEGORY.value:
                event_impact[ev].add(v)
            elif k == Relation.MENTIONS_ORG.value:
                cat = tweet_category.get(u, "")
                cat_label = cat.split(":", 1)[1] if ":" in cat else ""
                if cat_label in HUMANITARIAN_RESPONSE_CATEGORIES:
                    org_event.add((v, ev))

        for ev, locs in event_locs.items():
            for loc in locs:
                self.g.add_edge(ev, loc, key=Relation.OCCURRED_AT.value,
                                predicate=Relation.OCCURRED_AT.value, inferred=True)
        for ev, cats in event_impact.items():
            for cat in cats:
                self.g.add_edge(ev, cat, key=Relation.HAS_IMPACT.value,
                                predicate=Relation.HAS_IMPACT.value, inferred=True)
        for org, ev in org_event:
            self.g.add_edge(org, ev, key=Relation.RESPONDS_TO.value,
                            predicate=Relation.RESPONDS_TO.value, inferred=True)

        logger.info("Inferred edges: %d OCCURRED_AT, %d HAS_IMPACT, %d RESPONDS_TO",
                    sum(len(v) for v in event_locs.values()),
                    sum(len(v) for v in event_impact.values()),
                    len(org_event))

    # ── statistics ────────────────────────────────────────────

    def stats(self) -> Dict:
        node_types = Counter(d["type"] for _, d in self.g.nodes(data=True))
        edge_types = Counter(k for _, _, k in self.g.edges(keys=True))
        return {
            "num_nodes": self.g.number_of_nodes(),
            "num_edges": self.g.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "density": round(nx.density(self.g), 6),
        }

    # ── exports ───────────────────────────────────────────────

    def export_all(self, out_dir: str, name: str = "disaster_kg") -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "graphml": str(out / f"{name}.graphml"),
            "json":    str(out / f"{name}.json"),
            "cypher":  str(out / f"{name}.cypher"),
            "stats":   str(out / f"{name}.stats.json"),
        }
        self._export_graphml(paths["graphml"])
        self._export_json(paths["json"])
        self._export_cypher(paths["cypher"])
        with open(paths["stats"], "w") as f:
            json.dump(self.stats(), f, indent=2)
        logger.info("Exported KG to %s", out)
        return paths

    def _export_graphml(self, path: str) -> None:
        # GraphML can't store the multigraph 'key'; collapse attrs to strings.
        h = nx.MultiDiGraph()
        for n, d in self.g.nodes(data=True):
            h.add_node(n, **{k: ("" if v is None else str(v)) for k, v in d.items()})
        for u, v, k, d in self.g.edges(keys=True, data=True):
            h.add_edge(u, v, key=k,
                       **{kk: ("" if vv is None else str(vv)) for kk, vv in d.items()})
        nx.write_graphml(h, path)

    def _export_json(self, path: str) -> None:
        # networkx <3.4 doesn't accept the `edges` kwarg; call defensively so the
        # export works across versions.
        try:
            data = nx.node_link_data(self.g, edges="links")
        except TypeError:
            data = nx.node_link_data(self.g)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_cypher(self, path: str) -> None:
        """Emit MERGE statements to rebuild the graph in Neo4j."""
        lines = ["// Disaster Knowledge Graph — generated load script",
                 "// Run with: cypher-shell < disaster_kg.cypher", ""]
        for n, d in self.g.nodes(data=True):
            label = d["type"]
            props = {
                "id":    n,
                "name":  d.get("label", ""),
            }
            if d.get("text"):
                props["text"] = d["text"]
            prop_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
            lines.append(f"MERGE (:{label} {{{prop_str}}});")
        lines.append("")
        for u, v, k in self.g.edges(keys=True):
            lines.append(
                f'MATCH (a {{id: {json.dumps(u)}}}), (b {{id: {json.dumps(v)}}}) '
                f'MERGE (a)-[:{k}]->(b);'
            )
        with open(path, "w") as f:
            f.write("\n".join(lines))
