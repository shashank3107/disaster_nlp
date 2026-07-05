"""
Stage 5 — Visualisation.

Renders the knowledge graph with matplotlib (Agg backend, HPC-safe). Tweet nodes
are numerous and visually noisy, so two views are produced:

  * full      — every node (good for a "look how big it is" figure)
  * schema    — Tweet nodes collapsed away, showing only the semantic backbone
                (Event / Category / Severity / Location / Organization), which is
                the figure you actually want in a thesis.

A per-event subgraph can also be rendered to inspect one disaster at a time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from .schema import EntityType, NODE_STYLE

logger = logging.getLogger(__name__)

_TWEET = EntityType.TWEET.value


def _legend(types_present):
    return [
        mpatches.Patch(color=NODE_STYLE[t]["color"], label=t)
        for t in NODE_STYLE if t in types_present
    ]


def _draw(g: nx.MultiDiGraph, title: str, path: str, seed: int = 42,
          with_labels: bool = True, label_types=None) -> None:
    if g.number_of_nodes() == 0:
        logger.warning("Nothing to draw for %s", title)
        return

    colors = [d.get("color", "#cccccc") for _, d in g.nodes(data=True)]
    sizes  = [d.get("size", 300)        for _, d in g.nodes(data=True)]
    types_present = {d.get("type") for _, d in g.nodes(data=True)}

    pos = nx.spring_layout(g, seed=seed, k=0.6, iterations=60)

    plt.figure(figsize=(18, 13))
    nx.draw_networkx_edges(g, pos, alpha=0.25, width=0.7, arrows=False)
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=sizes,
                           alpha=0.9, linewidths=0.3, edgecolors="white")

    if with_labels:
        if label_types is None:
            labels = {n: d.get("label", "") for n, d in g.nodes(data=True)}
        else:
            labels = {n: d.get("label", "")
                      for n, d in g.nodes(data=True) if d.get("type") in label_types}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    plt.legend(handles=_legend(types_present), loc="upper left", fontsize=11)
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure -> %s", path)


def schema_subgraph(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Drop Tweet nodes; keep the semantic backbone via inferred/aggregated edges."""
    keep = [n for n, d in g.nodes(data=True) if d.get("type") != _TWEET]
    return g.subgraph(keep).copy()


def event_subgraph(g: nx.MultiDiGraph, event_node: str, radius: int = 2) -> nx.MultiDiGraph:
    """Neighbourhood around a single Event node."""
    if not g.has_node(event_node):
        return nx.MultiDiGraph()
    nodes = nx.ego_graph(g.to_undirected(as_view=True), event_node, radius=radius).nodes()
    return g.subgraph(nodes).copy()


# Node types always kept in the core view (the semantic skeleton).
_ALWAYS_KEEP = {
    EntityType.EVENT.value,
    EntityType.HUMANITARIAN_CATEGORY.value,
    EntityType.DAMAGE_SEVERITY.value,
}


def core_subgraph(g: nx.MultiDiGraph, min_degree: int = 3) -> nx.MultiDiGraph:
    """
    The clean 'backbone' figure: drop Tweet nodes and the long tail of
    single-mention entities, keeping Events, categories, and only the
    frequently-mentioned Locations / Organizations / Persons / Hashtags
    (degree >= min_degree). This is the readable, publication-style view.
    """
    undirected_deg = dict(g.degree())
    keep = []
    for n, d in g.nodes(data=True):
        t = d.get("type")
        if t == _TWEET:
            continue
        if t in _ALWAYS_KEEP or undirected_deg.get(n, 0) >= min_degree:
            keep.append(n)
    return g.subgraph(keep).copy()


def render(g: nx.MultiDiGraph, out_dir: str, prefix: str = "kg") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Core backbone (the clean thesis figure) — Events + categories + frequent
    # entities only, all labelled.
    cg = core_subgraph(g, min_degree=3)
    _draw(cg, "Disaster Knowledge Graph — Core Backbone (degree ≥ 3 entities)",
          str(out / f"{prefix}_core.png"), with_labels=True)

    # Schema backbone (all entities, tweets collapsed) — shows full scale.
    sg = schema_subgraph(g)
    _draw(sg, "Disaster Knowledge Graph — Semantic Backbone (tweets collapsed)",
          str(out / f"{prefix}_schema.png"), with_labels=True)

    # Full graph — label only the high-level nodes to avoid clutter.
    label_types = {t for t in NODE_STYLE if t != _TWEET}
    _draw(g, "Disaster Knowledge Graph — Full (all tweets & entities)",
          str(out / f"{prefix}_full.png"), with_labels=True, label_types=label_types)

    # One figure per event.
    events = [n for n, d in g.nodes(data=True) if d.get("type") == EntityType.EVENT.value]
    for ev in events:
        eg = event_subgraph(g, ev, radius=2)
        eg = schema_subgraph(eg)   # keep event view readable
        safe = ev.split(":", 1)[1].replace(" ", "_")
        _draw(eg, f"Event subgraph — {safe}",
              str(out / f"{prefix}_event_{safe}.png"),
              with_labels=True)
