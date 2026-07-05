#!/usr/bin/env python
"""
Disaster Tweet Intelligence — Streamlit application.

Pipeline (all visible in the UI):
    1. Upload a CSV of tweets for an event
    2. Classify  — RoBERTa checkpoints predict informative / humanitarian / damage
    3. Knowledge graph — BERT-NER -> relations -> triples -> graph (each step shown)
    4. Chatbot — hybrid Graph-RAG retrieval + Gemini-API grounded answers

Run locally:
    streamlit run app.py

See APP_README.md for setup (checkpoints, env, Gemini key).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ── project imports ───────────────────────────────────────────
from src.app.ingest import (
    load_csv, validate_csv, base_records, meta_map, classify_records,
    classified_dataframe, validate_classified, classified_records,
    load_news_csv, validate_news, news_records,
    REQUIRED_COLUMNS, TASKS,
)
from src.app.translate import Translator, translate_records
from src.app.geo import geocode_many
from src.app.gemini_generator import GeminiGenerator
from src.app.resource_extraction import (
    ResourceExtractor, resource_dataframe, match_need_availability, unmet_alerts,
)
from src.kg.ner import EntityExtractor
from src.kg.schema import Entity, EntityType
from src.kg.relations import extract_triples
from src.kg.graph_builder import KnowledgeGraph
from src.kg import visualize_kg
from src.kg.export_html import export_html
from src.rag.kg_store import KGStore
from src.rag.embed import VectorIndex
from src.rag.query_understanding import QueryUnderstanding
from src.rag.retriever import HybridRetriever


st.set_page_config(page_title="Disaster Tweet Intelligence", page_icon="🌀", layout="wide")


# ── cached heavy resources (loaded once per session) ──────────

@st.cache_resource(show_spinner=False)
def get_classifier(ckpt_dir: str, device: str | None):
    from src.inference import DisasterInference
    return DisasterInference(ckpt_dir, device=device)

@st.cache_resource(show_spinner=False)
def get_ner(ner_model: str, use_model: bool, device: str | None):
    return EntityExtractor(model_name=ner_model, device=device, use_model=use_model)


def check_checkpoints(experiments_dir: str, model_key: str, tasks) -> list[str]:
    """Return a list of human-readable problems with the classifier checkpoints."""
    issues = []
    for t in tasks:
        d = Path(experiments_dir) / f"{model_key}_{t}" / "best"
        sf = d / "model.safetensors"
        if not d.exists():
            issues.append(f"`{t}`: folder missing → `{d}`")
        elif not sf.exists():
            issues.append(f"`{t}`: `model.safetensors` missing in `{d}`")
        elif sf.stat().st_size < 100_000_000:   # real file is ~498 MB
            issues.append(f"`{t}`: `model.safetensors` is only "
                          f"{sf.stat().st_size:,} bytes — likely truncated "
                          f"(should be ~498 MB). Re-copy it.")
    return issues


# ── session state ─────────────────────────────────────────────

def _init_state():
    for k, v in {
        "records": None, "df": None, "news_df": None, "kg": None, "store": None,
        "index": None, "retriever": None, "qu": None,
        "messages": [], "graph_html": None, "kg_stats": None,
        "resinfo": None, "resource_df": None,
    }.items():
        st.session_state.setdefault(k, v)

_init_state()


# ── sidebar config ────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    event_name = st.text_input("Event name", value="my_event",
                               help="Used as the central Event node in the graph")
    st.subheader("Gemini (chatbot)")
    api_key = st.text_input("Gemini API key", type="password",
                            help="Or set GEMINI_API_KEY in your environment")
    gemini_model = st.text_input("Gemini model", value="gemini-2.0-flash")

    st.subheader("Models")
    model_key = st.selectbox("Classifier family", ["roberta", "distilbert"], index=0)
    experiments_dir = st.text_input("Checkpoints dir", value="./experiments")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    use_ner_model = st.checkbox("Use BERT-NER (off = regex fallback)", value=True)
    ner_model = "dslim/bert-base-NER"
    translate_non_english = st.checkbox(
        "Translate non-English tweets → English", value=True,
        help="Detects Indian-language tweets (native scripts + romanized Hinglish) "
             "and translates them to English before classification.")

    st.subheader("Retrieval")
    only_informative = st.checkbox("Graph: only informative tweets", value=True)
    top_k = st.slider("Tweets sent to the LLM (top-k)", 3, 15, 8)
    sem_k = st.slider("Semantic candidates", 10, 50, 20)

st.title("🌀 Disaster Tweet Intelligence")
st.caption("Classification → Knowledge Graph → Graph-RAG chatbot")

tab_upload, tab_kg, tab_chat, tab_dash = st.tabs(
    ["① Upload & Classify", "② Knowledge Graph", "③ Chatbot", "④ Resource Dashboard"]
)


# ════════════════════════════════════════════════════════════
# TAB 1 — Upload & Classify
# ════════════════════════════════════════════════════════════

with tab_upload:
    st.subheader("Upload data")
    st.caption("Upload tweets and/or news articles — both are classified together. "
               "News articles are sliced into tweet-sized snippets first.")

    col_tw, col_news = st.columns(2)
    with col_tw:
        st.markdown("**🐦 Tweets**")
        st.caption("Columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
        up = st.file_uploader("Tweets CSV/TSV", type=["csv", "tsv", "tab"],
                              key="tweet_uploader")
    with col_news:
        st.markdown("**📰 News articles**")
        st.caption("Scraped articles (needs `full_text`/`snippet`/`title`).")
        nup = st.file_uploader("News CSV/TSV", type=["csv", "tsv", "tab"],
                               key="news_uploader")
        max_chunks = st.slider("Max snippets per article", 3, 40, 15)

    with st.expander("…or load previously classified data (skip classification)"):
        st.caption("Expects columns: tweet_id, tweet_text, text_info, text_human, "
                   "text_damage — i.e. a file saved with the button below.")
        cup = st.file_uploader("Classified CSV/TSV", type=["csv", "tsv", "tab"],
                               key="classified_uploader")
        if cup is not None:
            cdf_in = load_csv(cup)
            ok, msg = validate_classified(cdf_in)
            if not ok:
                st.error(msg)
            elif st.button("📥 Load classified data"):
                st.session_state.df = None
                st.session_state.records = classified_records(cdf_in, event_name)
                st.session_state.kg = None     # invalidate downstream
                st.session_state.store = None
                st.success(f"Loaded {len(st.session_state.records)} classified tweets. "
                           "Open tab ② to build the knowledge graph.")

    # ── handle tweet upload ──
    if up is not None:
        df = load_csv(up)
        ok, msg = validate_csv(df)
        if not ok:
            st.session_state.df = None
            st.error(msg)
        else:
            st.session_state.df = df
            st.success(f"Loaded {len(df)} tweets.")
            st.dataframe(df.head(5), use_container_width=True)

    # ── handle news upload ──
    if nup is not None:
        ndf = load_news_csv(nup)
        ok, msg = validate_news(ndf)
        if not ok:
            st.session_state.news_df = None
            st.error(msg)
        else:
            st.session_state.news_df = ndf
            n_snip = len(news_records(ndf, event_name, max_chunks_per_article=max_chunks))
            st.success(f"Loaded {len(ndf)} news articles → ~{n_snip} tweet-sized snippets.")
            prev_cols = [c for c in ("title", "source", "scrape_status")
                         if c in ndf.columns]
            if prev_cols:
                st.dataframe(ndf[prev_cols].head(5), use_container_width=True)

    # ── combined classification (tweets + news together) ──
    have_tw   = st.session_state.df is not None
    have_news = st.session_state.news_df is not None
    if have_tw or have_news:
        st.divider()

        # Pre-flight: are the classifier checkpoints present and intact?
        ckpt_issues = check_checkpoints(experiments_dir, model_key, TASKS)
        if ckpt_issues:
            st.error("Classifier checkpoints not ready — classification would "
                     "return empty labels. Fix these, then re-run:")
            for issue in ckpt_issues:
                st.write("• " + issue)

        if st.button("🚀 Run classification", type="primary", disabled=bool(ckpt_issues)):
            records = []
            if have_tw:
                records += base_records(st.session_state.df, event_name)
            if have_news:
                records += news_records(st.session_state.news_df, event_name,
                                        max_chunks_per_article=max_chunks)

            if not records:
                st.error("Nothing to classify — the uploaded file(s) produced no text.")
                st.stop()

            st.caption(f"Classifying {len(records)} items "
                       f"({sum(r['tweet_id'].startswith('news-') for r in records)} "
                       "from news, rest from tweets).")
            prog = st.progress(0.0, text="starting…")

            def cb(label, frac):
                prog.progress(min(frac, 1.0), text=label)

            # Translate non-English text to English first (classifiers are
            # English-only). Original text is kept on each record.
            if translate_non_english:
                translator = Translator()
                if not translator.available():
                    st.warning("Translation requested but `deep-translator` isn't "
                               "installed — classifying text as-is.")
                else:
                    with st.spinner("Detecting languages & translating to English…"):
                        tx = translate_records(records, translator, progress_cb=cb)
                    if tx["translated"]:
                        st.info(f"Translated {tx['translated']} non-English item(s) "
                                f"to English before classification.")
                    else:
                        st.caption("No non-English text detected.")

            try:
                with st.spinner("Loading checkpoints & classifying…"):
                    load_clf = lambda ckpt: get_classifier(ckpt, device)
                    records = classify_records(
                        records, experiments_dir=experiments_dir,
                        model_key=model_key, device=device,
                        load_clf=load_clf, progress_cb=cb,
                    )
                prog.progress(1.0, text="done")
            except Exception as e:
                st.error(f"Classification failed while loading a model: {e}")
                st.stop()

            # Warn if any task came back entirely empty (silent skip / load issue)
            empties = [t for t in TASKS if all(not r.get(t) for r in records)]
            if empties:
                st.warning(f"These tasks produced no labels: {empties}. "
                           f"Check that experiments/{model_key}_<task>/best exists "
                           f"and model.safetensors is the full ~498 MB file.")

            st.session_state.records = records
            st.session_state.kg = None   # invalidate downstream
            st.session_state.store = None

    # show classification results if present
    if st.session_state.records:
        recs = st.session_state.records
        rdf = pd.DataFrame(recs)
        st.markdown("### Classification results")
        if "translated" in rdf.columns and rdf["translated"].any():
            st.caption(f"{int(rdf['translated'].sum())} tweet(s) were translated to "
                       "English — `text` shows the translation, `text_original` the source.")
        show_cols = ["tweet_id", "lang", "text", "text_original",
                     "informative", "humanitarian", "damage"]
        st.dataframe(rdf[[c for c in show_cols if c in rdf.columns]],
                     use_container_width=True, height=320)

        # save classified data in the compact "tweet" format
        cdf = classified_dataframe(recs)
        st.download_button(
            "⬇️ Save classified data (CSV)",
            data=cdf.to_csv(index=False),
            file_name=f"classified_{event_name}.csv", mime="text/csv",
            help="Columns: tweet_id, tweet_text, text_info, text_human, text_damage",
        )

        c1, c2, c3 = st.columns(3)
        for col, task in zip((c1, c2, c3), TASKS):
            if task in rdf.columns:
                with col:
                    st.markdown(f"**{task}**")
                    st.bar_chart(rdf[task].value_counts())


# ════════════════════════════════════════════════════════════
# TAB 2 — Knowledge Graph (steps visible)
# ════════════════════════════════════════════════════════════

def build_knowledge_graph(records, meta, use_ner_model, ner_model, device,
                          only_informative):
    """Run NER → relations → graph, surfacing each step in the UI."""
    # filter
    if only_informative:
        kept = [r for r in records if r.get("informative", "") == "informative"]
        if not kept:   # fall back so the demo never ends up empty
            kept = records
    else:
        kept = records

    # Stage: NER
    with st.status("Stage 1 — Named Entity Recognition (BERT-NER)…", expanded=True) as s:
        extractor = get_ner(ner_model, use_ner_model, device)
        texts = [r["text"] for r in kept]
        ent_lists = extractor.extract_batch(texts, batch_size=64)
        n_ent = sum(len(e) for e in ent_lists)
        st.write(f"Extracted **{n_ent}** entities from **{len(kept)}** tweets "
                 f"({n_ent / max(len(kept),1):.2f}/tweet).")
        sample = [f"{e.text} ({e.type.value})" for el in ent_lists[:30] for e in el][:12]
        st.caption("Sample: " + ", ".join(sample))
        s.update(label=f"Stage 1 — NER ✓ ({n_ent} entities)", state="complete")

    # Stage: relations → triples
    with st.status("Stage 2 — Relation extraction → triples…", expanded=True) as s:
        all_triples = []
        for rec, ents in zip(kept, ent_lists):
            entities = [Entity(text=e.text, type=e.type, norm=e.norm, score=e.score)
                        for e in ents]
            all_triples.extend(extract_triples(rec, entities))
        st.write(f"Generated **{len(all_triples)}** subject–predicate–object triples.")
        ex = all_triples[:6]
        st.caption("Examples: " + " · ".join(
            f"{t.subj.split(':',1)[-1]} —{t.pred.value}→ {t.obj.split(':',1)[-1]}"
            for t in ex))
        s.update(label=f"Stage 2 — Triples ✓ ({len(all_triples)})", state="complete")

    # Stage: assemble graph
    with st.status("Stage 3 — Graph assembly + entity alignment…", expanded=True) as s:
        kg = KnowledgeGraph()
        texts_map = {r["tweet_id"]: r["text"] for r in kept}
        kg.add_triples(all_triples, texts=texts_map, meta=meta)
        stats = kg.stats()
        st.write(f"Graph built: **{stats['num_nodes']} nodes**, "
                 f"**{stats['num_edges']} edges**.")
        st.json(stats, expanded=False)
        s.update(label=f"Stage 3 — Graph ✓ ({stats['num_nodes']}n/{stats['num_edges']}e)",
                 state="complete")

    # Stage: index for RAG
    with st.status("Stage 4 — Building RAG vector index (BGE + FAISS)…", expanded=True) as s:
        store = KGStore(kg.g)
        index = VectorIndex()
        index.build(store)
        qu = QueryUnderstanding(store, use_ner=use_ner_model, ner_model=ner_model,
                                device=device)
        retr = HybridRetriever(store, index)
        st.write(f"Indexed **{len(index.ids)}** tweets for semantic retrieval.")
        s.update(label=f"Stage 4 — RAG index ✓ ({len(index.ids)} tweets)", state="complete")

    # interactive HTML
    core = visualize_kg.core_subgraph(kg.g, min_degree=2)
    tmp = Path(tempfile.gettempdir()) / "kg_app_core.html"
    export_html(core, str(tmp), title=f"KG — {event_name}", cdn_resources="remote")
    html = tmp.read_text()

    return kg, store, index, qu, retr, stats, html


with tab_kg:
    if not st.session_state.records:
        st.info("Run classification in tab ① first.")
    else:
        if st.button("🕸️ Build knowledge graph", type="primary"):
            meta = meta_map(st.session_state.records)
            kg, store, index, qu, retr, stats, html = build_knowledge_graph(
                st.session_state.records, meta, use_ner_model, ner_model, device,
                only_informative,
            )
            st.session_state.update(
                kg=kg, store=store, index=index, qu=qu, retriever=retr,
                kg_stats=stats, graph_html=html,
            )
            st.success("Knowledge graph ready. Open tab ③ to chat with it.")

        if st.session_state.graph_html:
            st.markdown("### Interactive knowledge graph")
            st.caption("Drag • zoom • hover. (Core view: events + frequent entities.)")
            st.components.v1.html(st.session_state.graph_html, height=650, scrolling=True)
            st.download_button("⬇️ Download graph HTML",
                               data=st.session_state.graph_html,
                               file_name=f"kg_{event_name}.html", mime="text/html")


# ════════════════════════════════════════════════════════════
# TAB 3 — Chatbot (Graph-RAG + Gemini)
# ════════════════════════════════════════════════════════════

with tab_chat:
    if not st.session_state.retriever:
        st.info("Build the knowledge graph in tab ② first.")
    else:
        gen = GeminiGenerator(api_key=api_key, model=gemini_model)
        if not gen.available():
            st.warning("No Gemini API key — answers will show retrieved tweets only. "
                       "Add a key in the sidebar to enable generated answers.")

        # render history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        q = st.chat_input("Ask about this event…")
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                spec = st.session_state.qu.parse(q)
                hits = st.session_state.retriever.retrieve(
                    q, spec, top_k=top_k, sem_k=sem_k)

                with st.expander(f"🔎 Retrieved {len(hits)} tweets  "
                                 f"(filters: {spec.describe()})"):
                    for i, r in enumerate(hits, 1):
                        d = r.doc
                        st.markdown(
                            f"**[{i}]** `{d.tweet_id}` · {d.event or 'n/a'}"
                            f"{' · ' + d.category if d.category else ''}  \n"
                            f"_{d.text}_  \n"
                            f"<span style='color:#888'>via {r.how}, score={r.score}</span>",
                            unsafe_allow_html=True,
                        )

                if gen.available():
                    with st.spinner("Generating grounded answer…"):
                        answer = gen.generate(q, hits)
                else:
                    answer = "**(no LLM key) Top retrieved tweets:**\n\n" + "\n".join(
                        f"- [{r.doc.tweet_id}] {r.doc.text}" for r in hits[:5])
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


# ════════════════════════════════════════════════════════════
# TAB 4 — Resource Dashboard (need / availability + allocation)
# ════════════════════════════════════════════════════════════

with tab_dash:
    if not st.session_state.records:
        st.info("Run classification in tab ① first.")
    else:
        st.subheader("Resource needs & availability")
        st.caption("Extracts what resource each tweet is about (food/water/shelter/"
                   "medical/…) and whether it is a NEED or an AVAILABILITY, then "
                   "surfaces trends, alerts, and allocation suggestions.")

        extractor_mode = ("Gemini" if api_key else "keyword fallback (no API key)")
        if st.button(f"📦 Extract resources & build dashboard  ·  using {extractor_mode}",
                     type="primary"):
            prog = st.progress(0.0, text="starting…")
            extractor = ResourceExtractor(api_key=api_key, model=gemini_model,
                                          use_llm=bool(api_key))
            resinfo = extractor.extract(
                st.session_state.records,
                progress_cb=lambda lbl, frac: prog.progress(min(frac, 1.0), text=lbl),
            )
            prog.progress(1.0, text="done")
            st.session_state.resinfo = resinfo
            st.session_state.resource_df = resource_dataframe(
                st.session_state.records, resinfo)

    rdf = st.session_state.resource_df
    if rdf is not None and not rdf.empty:
        need_df  = rdf[rdf["stance"] == "need"]
        avail_df = rdf[rdf["stance"] == "availability"]

        # ── KPI row ──
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tweets analysed", len(st.session_state.records))
        k2.metric("Need mentions", len(need_df))
        k3.metric("Availability mentions", len(avail_df))
        k4.metric("Locations", rdf.loc[rdf["location"] != "unknown", "location"].nunique())

        # ── Alerts (unmet needs) ──
        st.markdown("### 🚨 Priority alerts — unmet needs")
        match = match_need_availability(rdf)
        alerts = unmet_alerts(match)
        if alerts.empty:
            st.success("No unmet-need hotspots detected (every need has some availability).")
        else:
            for _, row in alerts.head(12).iterrows():
                loc = row["location"] if row["location"] != "unknown" else "location unspecified"
                st.error(f"**{row['resource'].title()}** at **{loc}** — "
                         f"{int(row['need'])} need vs {int(row['availability'])} available "
                         f"(gap {int(row['gap'])})")

        # ── Affected-areas map ──
        st.markdown("### 🗺️ Affected-areas map")
        st.caption("Each circle is a location with reported resource activity. "
                   "Circle size = number of need mentions; red = unmet need "
                   "(need > availability), orange = needs being met, green = "
                   "availability only.")
        if st.checkbox("Show affected-areas map", value=True):
            import pydeck as pdk

            # Aggregate need / availability per location.
            geo_src = rdf[rdf["location"] != "unknown"]
            if geo_src.empty:
                st.info("No place names were extracted from the tweets, so there's "
                        "nothing to map. (Location extraction needs a Gemini API key.)")
            else:
                loc_stats = (geo_src
                             .pivot_table(index="location", columns="stance",
                                          values="tweet_id", aggfunc="count",
                                          fill_value=0)
                             .reset_index())
                for col in ("need", "availability"):
                    if col not in loc_stats.columns:
                        loc_stats[col] = 0
                loc_stats["gap"] = loc_stats["need"] - loc_stats["availability"]

                with st.spinner("Geocoding locations…"):
                    coords = geocode_many(loc_stats["location"].tolist())

                loc_stats["lat"] = loc_stats["location"].map(
                    lambda n: coords.get(n, (None, None))[0])
                loc_stats["lon"] = loc_stats["location"].map(
                    lambda n: coords.get(n, (None, None))[1])
                map_df = loc_stats.dropna(subset=["lat", "lon"]).copy()

                missing = sorted(set(loc_stats["location"]) - set(map_df["location"]))
                if map_df.empty:
                    st.warning("Could not geocode any of the extracted locations: "
                               + ", ".join(missing[:20]))
                else:
                    # Circle radius (metres) scales with need volume; colour by gap.
                    map_df["radius"] = 20000 + map_df["need"] * 12000

                    def _color(row):
                        if row["gap"] > 0:        # unmet need
                            return [220, 30, 30, 160]
                        if row["need"] > 0:       # need present but met
                            return [255, 140, 0, 160]
                        return [40, 160, 60, 160]  # availability only
                    map_df["color"] = map_df.apply(_color, axis=1)

                    layer = pdk.Layer(
                        "ScatterplotLayer", data=map_df,
                        get_position="[lon, lat]", get_radius="radius",
                        get_fill_color="color", pickable=True, opacity=0.6,
                        stroked=True, get_line_color=[60, 60, 60], line_width_min_pixels=1,
                    )
                    view = pdk.ViewState(
                        latitude=float(map_df["lat"].mean()),
                        longitude=float(map_df["lon"].mean()),
                        zoom=3.5, pitch=0,
                    )
                    tooltip = {"html": "<b>{location}</b><br/>"
                                       "Need: {need} · Availability: {availability}<br/>"
                                       "Gap: {gap}"}
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                             tooltip=tooltip,
                                             map_style="road"))
                    if missing:
                        st.caption("Not mapped (couldn't geocode): "
                                   + ", ".join(missing[:20]))

        # ── Resource breakdown ──
        st.markdown("### Resource breakdown (need vs availability)")
        by_res = (rdf[rdf["stance"].isin(["need", "availability"])]
                  .pivot_table(index="resource", columns="stance",
                               values="tweet_id", aggfunc="count", fill_value=0))
        if not by_res.empty:
            st.bar_chart(by_res)

        # ── Trend over time ──
        st.markdown("### Trend over time")
        ts = rdf.dropna(subset=["time"])
        if not ts.empty:
            series = (ts.assign(hour=ts["time"].dt.floor("h"))
                        .pivot_table(index="hour", columns="stance",
                                     values="tweet_id", aggfunc="count", fill_value=0))
            st.line_chart(series)
        else:
            st.caption("No parseable timestamps to plot a trend.")

        # ── Allocation table + locations ──
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Need ↔ availability by location")
            st.dataframe(match, use_container_width=True, height=320)
        with c2:
            st.markdown("### Top locations by need")
            if not need_df.empty:
                top_loc = (need_df[need_df["location"] != "unknown"]
                           .groupby("location")["tweet_id"].count()
                           .sort_values(ascending=False).head(12))
                if not top_loc.empty:
                    st.bar_chart(top_loc)
                else:
                    st.caption("No locations extracted (add a Gemini key for location extraction).")

        st.download_button("⬇️ Download resource table (CSV)",
                           data=rdf.to_csv(index=False),
                           file_name=f"resources_{event_name}.csv", mime="text/csv")
    elif st.session_state.records:
        st.caption("Click the button above to run resource extraction.")
