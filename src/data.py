"""
Data loading, preprocessing, and PyTorch Dataset for CrisisMMD.

Preprocessing philosophy (critical for research correctness):
  - Transformers tokenise subword units — aggressive cleaning destroys
    domain signals (e.g. "#earthquake" → "earthquake" loses the hashtag
    emphasis that BPE encodes differently).
  - We normalise only what adds noise and preserves what adds signal:
      * URLs removed (no semantic value after tokenisation)
      * Mentions normalised to @user (privacy + vocabulary reduction)
      * Hashtag symbol stripped BUT word kept (BPE handles #earthquake well,
        but some tokenisers split the # as a separate token causing issues)
      * HTML entities decoded
      * Extra whitespace collapsed
  - NO stopword removal, NO stemming, NO lowercasing (models are cased or
    handle casing via subword tokenisation).
"""

from __future__ import annotations

import re
import html
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split

from .config import TaskConfig, TrainConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_tweet(text: str) -> str:
    """
    Minimal, transformer-friendly tweet cleaning.

    Preserves:  hashtags (word only), mentions (@user), punctuation,
                disaster-domain vocabulary, numbers, emojis.
    Removes:    URLs, duplicate whitespace, HTML entities (decoded).
    """
    text = str(text)
    text = html.unescape(text)                          # &amp; → &
    text = re.sub(r"http\S+|www\.\S+", "", text)        # strip URLs
    text = re.sub(r"@\w+", "@user", text)               # anonymise mentions
    text = re.sub(r"#(\w+)", r"\1", text)               # #flood → flood
    text = re.sub(r"\s+", " ", text)                    # collapse whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────────
# TSV loader
# ─────────────────────────────────────────────────────────────

def load_split(
    filepath: str,
    task: TaskConfig,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load one TSV split, apply preprocessing, encode labels.

    Returns a DataFrame with columns: tweet_text, label (int).
    Rows with unknown labels (label not in task.label2id) are dropped
    with a warning — this handles dev/test sets that may contain
    label values absent from training.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Validate required columns
    for col in ("tweet_text", task.label_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' missing in {filepath}. "
                f"Available: {df.columns.tolist()}"
            )

    df = df[["tweet_text", task.label_col]].copy()
    df = df.rename(columns={task.label_col: "label_text"})
    df = df.dropna(subset=["tweet_text", "label_text"])
    df = df.drop_duplicates(subset=["tweet_text", "label_text"])

    # Preprocess text
    df["tweet_text"] = df["tweet_text"].apply(preprocess_tweet)
    df = df[df["tweet_text"].str.len() > 0]

    # Encode labels
    df["label"] = df["label_text"].map(task.label2id)
    unknown = df["label"].isna().sum()
    if unknown > 0:
        unknown_vals = df.loc[df["label"].isna(), "label_text"].unique().tolist()
        logger.warning(
            f"{unknown} rows with unknown labels dropped: {unknown_vals}"
        )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if n_samples is not None:
        df = df.sample(
            n=min(n_samples, len(df)), random_state=random_state
        ).reset_index(drop=True)

    return df.reset_index(drop=True)


def load_all_splits(
    data_dir: str,
    task: TaskConfig,
    n_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train / dev / test splits for a task."""
    base = Path(data_dir)
    train_df = load_split(str(base / task.train_file), task, n_samples)
    dev_df   = load_split(str(base / task.dev_file),   task)
    test_df  = load_split(str(base / task.test_file),  task)
    return train_df, dev_df, test_df


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Inverse-frequency class weights, capped to prevent instability.

    CrisisMMD is heavily imbalanced — e.g. in the humanitarian task,
    "missing_or_found_people" has ~3% of samples while "not_humanitarian"
    has ~40%.  Uncapped weights can cause loss spikes; we cap at max_weight.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)           # avoid div-by-zero
    total  = counts.sum()
    weights = total / (num_classes * counts)
    weights = np.clip(weights, 0.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def print_class_distribution(df: pd.DataFrame, task: TaskConfig, split: str) -> None:
    counts = df.groupby("label_text").size().sort_values(ascending=False)
    total  = len(df)
    logger.info(f"\n  [{split}] class distribution (n={total}):")
    for label, cnt in counts.items():
        bar = "█" * (cnt // max(total // 50, 1))
        logger.info(f"    {label:<50} {cnt:>5}  ({cnt/total*100:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    """
    Tokenise-on-init dataset.  We use dynamic padding via a DataCollator
    at the DataLoader level, so we store raw token IDs without padding here.

    Dynamic padding (vs fixed max_length padding) is important for efficiency:
    a batch of short tweets wastes no compute on padding tokens.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.labels    = labels
        self.encodings = tokenizer(
            texts,
            padding            = False,    # dynamic padding done by collator
            truncation         = True,
            max_length         = max_length,
            return_token_type_ids = False,  # not needed for RoBERTa/DeBERTa
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
