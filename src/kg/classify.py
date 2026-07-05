"""
Stage 1 — Classification.

Produces, for every tweet, the typed labels the rest of the KG hangs off:
    event_name, humanitarian_category, damage_severity, informative.

Two modes:

  * mode="gold"    (default) — read the labels already present in the CrisisMMD
    TSV columns. This makes the pipeline runnable with zero GPU and reproduces an
    "ideal classifier"; standard practice when the KG itself is the contribution.

  * mode="predict"           — load the fine-tuned checkpoints from ./experiments
    and predict the labels (the realistic end-to-end setting). Falls back to gold
    for any task whose checkpoint is missing.

Output: a list of record dicts, one per tweet, written as JSONL by the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column names in the CrisisMMD TSVs
COL_EVENT  = "event_name"
COL_ID     = "tweet_id"
COL_TEXT   = "tweet_text"

# Which checkpoint (under experiments/<model>_<task>/best) to use per task in
# predict mode, and which TSV column holds the gold label.
TASK_GOLD_COL = {
    "humanitarian": "label_text",
    "damage":       "label",
    "informative":  "label_text",
}


def load_tweets(tsv_path: str) -> pd.DataFrame:
    """Load a CrisisMMD TSV, de-duplicating on tweet_id (TSVs repeat per-image)."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    if COL_ID in df.columns:
        df = df.drop_duplicates(subset=[COL_ID]).reset_index(drop=True)
    logger.info("Loaded %d unique tweets from %s", len(df), tsv_path)
    return df


def detect_task(tsv_path: str) -> Optional[str]:
    """
    Infer which task a CrisisMMD TSV belongs to from its filename.

    This matters because every split shares generic column names ('label',
    'label_text') but the *vocabulary* differs: on the humanitarian split
    'label' holds humanitarian categories, on the damage split it holds damage
    severity. Reading the wrong column would pollute the graph with mistyped
    nodes, so in gold mode we only populate the TSV's own task.
    """
    name = Path(tsv_path).name.lower()
    for task in ("humanitarian", "damage", "informative"):
        if task in name:
            return task
    return None


def _gold_labels(df: pd.DataFrame, native_task: Optional[str]) -> Dict[str, List[str]]:
    """
    Pull the gold labels for the TSV's *own* task only.

    A single CrisisMMD split legitimately carries one task's labels. To attach a
    second task's labels to the same tweets, use predict mode (which runs each
    task's fine-tuned classifier over the raw text).
    """
    out: Dict[str, List[str]] = {}
    if native_task and native_task in TASK_GOLD_COL:
        col = TASK_GOLD_COL[native_task]
        if col in df.columns:
            out[native_task] = df[col].tolist()
    return out


def classify(
    tsv_path: str,
    mode: str = "gold",
    experiments_dir: str = "./experiments",
    model_key: str = "roberta",
    device: Optional[str] = None,
) -> List[Dict]:
    """
    Returns a list of records:
        {
          "tweet_id": ..., "event": ..., "text": ...,
          "humanitarian": <label or "">,
          "damage":       <label or "">,
          "informative":  <label or "">,
        }
    """
    df = load_tweets(tsv_path)
    n  = len(df)
    native_task = detect_task(tsv_path)
    logger.info("Native task for %s: %s", Path(tsv_path).name, native_task)

    records = [
        {
            "tweet_id":     df.at[i, COL_ID]    if COL_ID    in df.columns else str(i),
            "event":        df.at[i, COL_EVENT] if COL_EVENT in df.columns else "unknown_event",
            "text":         df.at[i, COL_TEXT]  if COL_TEXT  in df.columns else "",
            "humanitarian": "",
            "damage":       "",
            "informative":  "",
        }
        for i in range(n)
    ]

    # --- gold labels (always available as a baseline / fallback) ---
    gold = _gold_labels(df, native_task)
    # The single TSV we are reading only carries the label for its own task, so
    # in gold mode we populate whatever is present. Most useful inputs are the
    # humanitarian and damage TSVs.
    for task, labels in gold.items():
        for i, lab in enumerate(labels):
            records[i][task] = lab

    if mode == "gold":
        return records

    if mode != "predict":
        raise ValueError(f"Unknown classify mode: {mode!r} (use 'gold' or 'predict')")

    # --- predict mode: run the fine-tuned checkpoints over the raw text ---
    from ..inference import DisasterInference  # lazy import (torch heavy)

    texts = [r["text"] for r in records]
    for task in ("humanitarian", "damage", "informative"):
        ckpt = Path(experiments_dir) / f"{model_key}_{task}" / "best"
        if not ckpt.exists():
            logger.warning("No checkpoint for task '%s' at %s — keeping gold/empty.",
                           task, ckpt)
            continue
        logger.info("Predicting '%s' with %s", task, ckpt)
        clf   = DisasterInference(str(ckpt), device=device)
        preds = clf.predict_batch(texts, batch_size=64, preprocess=True)
        for i, p in enumerate(preds):
            records[i][task] = p["predicted"]
        del clf  # free GPU between tasks

    return records
