"""
Interactive HTML export for the disaster knowledge graph, using **pyvis**.

Produces a single .html file rendered with vis-network. Open it in any browser —
drag nodes, zoom, hover for tooltips, and use the physics controls. Node colour /
size encode the entity type; a floating legend is injected so the figure is
self-explanatory.

`cdn_resources`:
  * "remote"  (default) — small file; the vis-network library is fetched from a
    CDN on first open (needs internet once, then browser-cached).
  * "in_line"           — embeds the library in the file so it works fully offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from .schema import EntityType, NODE_STYLE

logger = logging.getLogger(__name__)

_TWEET = EntityType.TWEET.value


def _legend_html() -> str:
    """A small floating legend injected into the page body."""
    rows = []
    for t, s in NODE_STYLE.items():
        rows.append(
            f'<div style="margin:2px 0">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{s["color"]};margin-right:6px;vertical-align:middle"></span>{t}</div>'
        )
    return (
        '<div style="position:absolute;top:12px;left:12px;z-index:999;'
        'background:rgba(255,255,255,.94);border:1px solid #ddd;border-radius:8px;'
        'padding:10px 12px;font:13px system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.12)">'
        '<b>Disaster Knowledge Graph</b>'
        + "".join(rows)
        + '<div style="color:#666;font-size:12px;margin-top:6px">'
          'drag • zoom • hover for details</div></div>'
    )


def export_html(
    g: nx.MultiDiGraph,
    path: str,
    title: str = "Disaster Knowledge Graph",
    cdn_resources: str = "remote",
    notebook: bool = False,
) -> str:
    """Write an interactive pyvis HTML file for graph `g`."""
    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        bgcolor="#fafafa",
        font_color="#222",
        notebook=notebook,
        cdn_resources=cdn_resources,
    )

    # Nodes — size scaled from the schema's size hint; tooltip carries type/text.
    for n, d in g.nodes(data=True):
        ntype = d.get("type", "")
        label = d.get("label", "")
        tip   = f"{ntype}: {label}"
        if d.get("text"):
            tip += f"\n{d['text'][:200]}"
        net.add_node(
            n,
            label=label,
            title=tip,
            color=d.get("color", "#cccccc"),
            value=float(d.get("size", 300)),
            shape="dot",
            font={"size": 22 if ntype == EntityType.EVENT.value else 12},
        )

    # Edges — collapse parallel edges, keep the predicate as hover title.
    seen = set()
    for u, v, k in g.edges(keys=True):
        if (u, v, k) in seen:
            continue
        seen.add((u, v, k))
        net.add_edge(u, v, title=k, color="#bbbbbb", arrows="to")

    # Physics tuned for a hub-and-spoke disaster graph; show the control buttons.
    net.barnes_hut(gravity=-9000, central_gravity=0.3, spring_length=130,
                   spring_strength=0.02, damping=0.4)
    net.show_buttons(filter_=["physics"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # pyvis 0.3.x: generate_html() then inject our legend before writing.
    html = net.generate_html(notebook=notebook)
    html = html.replace("<body>", "<body>\n" + _legend_html(), 1)
    with open(path, "w") as f:
        f.write(html)

    logger.info("Wrote interactive HTML (%d nodes, %d edges) -> %s",
                g.number_of_nodes(), g.number_of_edges(), path)
    return path
