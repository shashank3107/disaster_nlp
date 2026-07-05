"""
CSV ingestion + classification for the app.

Input CSV columns (one event per file):
    tweet_url, tweet_id, username, tweet_text, timestamp

Unlike the CrisisMMD TSVs, these tweets have NO gold labels, so we always run the
fine-tuned checkpoints (predict mode) for all three tasks. RoBERTa-base is used by
default — it was the strongest model in our benchmark across tasks.

Returns per-tweet records compatible with the KG pipeline, plus a `meta` map
(url/username/timestamp) for provenance on Tweet nodes.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["tweet_url", "tweet_id", "username", "tweet_text", "timestamp"]
TASKS = ("informative", "humanitarian", "damage")

# News-article CSV (scraped articles). We only hard-require a text source; the
# rest are used for provenance when present.
NEWS_TEXT_COLUMNS = ["full_text", "snippet", "title"]
TWEET_MAX_LEN = 280   # target length when slicing an article into tweet-sized texts

# Saved classified-data format: a compact "tweet" table that keeps the predicted
# labels.  Column <-> record-key mapping (label keys mirror TASKS).
CLASSIFIED_COLUMNS = ["tweet_id", "tweet_text", "text_info", "text_human", "text_damage"]
_CLASSIFIED_TO_RECORD = {
    "tweet_id":    "tweet_id",
    "tweet_text":  "text",
    "text_info":   "informative",
    "text_human":  "humanitarian",
    "text_damage": "damage",
}


def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV is missing required columns: {missing}"
    if df.empty:
        return False, "CSV has no rows."
    return True, "ok"


def load_csv(path_or_buffer) -> pd.DataFrame:
    # Pick the delimiter from the file name when available; TSV files use tabs.
    name = getattr(path_or_buffer, "name", path_or_buffer)
    sep = "\t" if isinstance(name, str) and name.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path_or_buffer, sep=sep, dtype=str).fillna("")
    df = df.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)
    return df


def base_records(df: pd.DataFrame, event_name: str) -> List[Dict]:
    """Skeleton records (no labels yet) from the uploaded CSV."""
    ev = (event_name or "uploaded_event").strip()
    out = []
    for _, r in df.iterrows():
        out.append({
            "tweet_id":     str(r["tweet_id"]),
            "event":        ev,
            "text":         str(r["tweet_text"]),
            "username":     str(r.get("username", "")),
            "url":          str(r.get("tweet_url", "")),
            "timestamp":    str(r.get("timestamp", "")),
            "informative":  "",
            "humanitarian": "",
            "damage":       "",
        })
    return out


# ── news-article ingestion → tweet-sized records ──────────────

def load_news_csv(path_or_buffer) -> pd.DataFrame:
    """Load a scraped-news CSV/TSV. Unlike tweets there's no tweet_id to dedup on,
    so we dedup on the article URL when available."""
    name = getattr(path_or_buffer, "name", path_or_buffer)
    sep = "\t" if isinstance(name, str) and name.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path_or_buffer, sep=sep, dtype=str).fillna("")
    if "article_url" in df.columns:
        df = df.drop_duplicates(subset=["article_url"]).reset_index(drop=True)
    return df


def validate_news(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "News file has no rows."
    if not any(c in df.columns for c in NEWS_TEXT_COLUMNS):
        return False, ("News file needs at least one text column: "
                       f"{NEWS_TEXT_COLUMNS}.")
    return True, "ok"


def _fix_mojibake(text: str) -> str:
    """Best-effort repair of UTF-8 text that was mis-decoded as Latin-1/CP1252
    (the tell-tale 'â', 'Â', '€' sequences seen in scraped articles)."""
    if any(m in text for m in ("Ã", "â", "Â", "€")):
        try:
            return text.encode("latin-1", "ignore").decode("utf-8", "ignore")
        except Exception:
            return text
    return text


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def _hard_split(sentence: str, max_len: int) -> List[str]:
    """Split an over-long sentence into <=max_len word-aligned pieces."""
    pieces, cur = [], ""
    for w in sentence.split():
        if len(cur) + len(w) + 1 > max_len and cur:
            pieces.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text: str, max_len: int = TWEET_MAX_LEN, min_len: int = 25) -> List[str]:
    """Slice free text into tweet-sized chunks on sentence boundaries.

    Sentences are greedily packed up to `max_len`; sentences longer than `max_len`
    are hard-split on word boundaries. Chunks shorter than `min_len` are dropped as
    boilerplate noise.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    chunks: List[str] = []
    cur = ""
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(s, max_len))
        elif len(cur) + len(s) + 1 > max_len:
            if cur:
                chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= min_len]


def news_records(df: pd.DataFrame, event_name: str,
                 max_len: int = TWEET_MAX_LEN,
                 max_chunks_per_article: int = 15) -> List[Dict]:
    """
    Turn scraped news articles into tweet-sized, KG-compatible records.

    Per article we pick the best available text (full_text when scraped ok, else
    snippet, else title), repair encoding artifacts, prepend the headline, and slice
    it into tweet-length chunks. Each chunk becomes a record indistinguishable from
    a real tweet, so it flows through the same classify → KG → dashboard pipeline.
    """
    ev = (event_name or "uploaded_event").strip()
    out: List[Dict] = []
    for i, r in df.iterrows():
        status = str(r.get("scrape_status", "")).strip().lower()
        full   = str(r.get("full_text", "")).strip()
        snip   = str(r.get("snippet", "")).strip()
        title  = str(r.get("title", "")).strip()

        if status == "ok" and full:
            content = full
        elif snip:
            content = snip
        elif title:
            content = title
        else:
            continue

        content = _fix_mojibake(content)
        title   = _fix_mojibake(title)

        chunks = chunk_text(content, max_len=max_len)
        # Lead with the headline — it's a concise, high-signal summary.
        if title and (not chunks or title.lower() not in chunks[0].lower()):
            chunks = [title] + chunks
        if max_chunks_per_article:
            chunks = chunks[:max_chunks_per_article]

        url    = str(r.get("article_url", ""))
        source = str(r.get("source", "")) or str(r.get("author", ""))
        ts     = str(r.get("published_date", ""))
        base   = hashlib.md5((url or title or str(i)).encode("utf-8")).hexdigest()[:10]

        for j, chunk in enumerate(chunks):
            out.append({
                "tweet_id":     f"news-{base}-{j}",
                "event":        ev,
                "text":         chunk,
                "username":     source,
                "url":          url,
                "timestamp":    ts,
                "informative":  "",
                "humanitarian": "",
                "damage":       "",
            })
    return out


def classified_dataframe(records: List[Dict]) -> pd.DataFrame:
    """Records -> compact 'tweet' table for saving: tweet_id, tweet_text,
    text_info, text_human, text_damage."""
    rows = [
        {col: str(r.get(key, "")) for col, key in _CLASSIFIED_TO_RECORD.items()}
        for r in records
    ]
    return pd.DataFrame(rows, columns=CLASSIFIED_COLUMNS)


def validate_classified(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in CLASSIFIED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Classified file is missing required columns: {missing}"
    if df.empty:
        return False, "Classified file has no rows."
    return True, "ok"


def classified_records(df: pd.DataFrame, event_name: str) -> List[Dict]:
    """Rebuild KG-compatible records from a saved classified table. Provenance
    (url/username/timestamp) isn't stored in this format, so it's left blank."""
    ev = (event_name or "uploaded_event").strip()
    out = []
    for _, r in df.iterrows():
        rec = {
            "tweet_id":     str(r["tweet_id"]),
            "event":        ev,
            "text":         str(r["tweet_text"]),
            "username":     "",
            "url":          "",
            "timestamp":    "",
        }
        for col, key in _CLASSIFIED_TO_RECORD.items():
            if key in TASKS:
                rec[key] = str(r.get(col, ""))
        out.append(rec)
    return out


def meta_map(records: List[Dict]) -> Dict[str, Dict]:
    """tweet_id -> {url, username, timestamp} for Tweet-node provenance."""
    return {
        r["tweet_id"]: {"url": r["url"], "username": r["username"],
                        "timestamp": r["timestamp"]}
        for r in records
    }


def classify_records(
    records: List[Dict],
    experiments_dir: str = "./experiments",
    model_key: str = "roberta",
    device: Optional[str] = None,
    tasks=TASKS,
    load_clf: Optional[Callable] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> List[Dict]:
    """
    Predict labels for all tasks and write them onto `records` in place.

    `load_clf(ckpt_dir)` is an injectable loader (the app passes a cached one so
    models aren't reloaded on every Streamlit rerun). Defaults to a fresh
    DisasterInference per task.
    """
    from src.inference import DisasterInference

    if load_clf is None:
        load_clf = lambda ckpt: DisasterInference(ckpt, device=device)

    texts = [r["text"] for r in records]
    for ti, task in enumerate(tasks):
        ckpt = Path(experiments_dir) / f"{model_key}_{task}" / "best"
        if not ckpt.exists():
            logger.warning("Missing checkpoint for %s at %s — skipping.", task, ckpt)
            if progress_cb:
                progress_cb(f"skipped {task} (no checkpoint)", (ti + 1) / len(tasks))
            continue
        if progress_cb:
            progress_cb(f"classifying: {task}", ti / len(tasks))
        clf = load_clf(str(ckpt))
        preds = clf.predict_batch(texts, batch_size=64, preprocess=True)
        for r, p in zip(records, preds):
            r[task] = p["predicted"]
            r[f"{task}_conf"] = p["confidence"]
        if progress_cb:
            progress_cb(f"done: {task}", (ti + 1) / len(tasks))
    return records
