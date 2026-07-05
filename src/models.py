"""
Model factory for DeBERTa-v3-base, RoBERTa-base, and DistilBERT.

All three models share the same HuggingFace AutoModelForSequenceClassification
interface, so the trainer code is 100% model-agnostic.

Why DeBERTa outperforms RoBERTa/BERT on disaster tweets:
  1. Disentangled Attention (DA): DeBERTa represents each token with two
     separate vectors — content and position — and computes attention using
     all four content×position cross-products.  This captures "flooded CITY"
     differently from "CITY flooded" even for short tweets where positional
     nuance matters.
  2. Enhanced Mask Decoder (EMD): Absolute positional embeddings are added
     only at the final softmax layer, preventing positional information from
     leaking into the pre-training objective in a way that would hurt
     fine-tuning on short texts.
  3. SentencePiece tokeniser with 128K vocabulary: rare disaster-domain terms
     (hurricane names, location abbreviations) receive fewer UNK/split tokens
     compared to the 50K GPT-2 BPE vocabulary of RoBERTa.
  4. v3 uses DeBERTaV2 architecture trained on 1.5× more data with improved
     loss objectives (replaced-token detection via ELECTRA-style training),
     making it extremely data-efficient for fine-tuning on ~14K tweets.

Why transformers outperform traditional ML (TF-IDF + SVM/LR):
  - Traditional: bag-of-words ignores word order, context, and polysemy.
    "The hospital is not damaged" and "The hospital is damaged" have nearly
    identical TF-IDF vectors despite opposite meanings.
  - Transformers: contextual embeddings capture the full sentence context.
    Bidirectional attention means each token representation is influenced by
    all other tokens.  Transfer learning from large pre-training corpora gives
    the model knowledge of disaster vocabulary it has never seen in fine-tuning.

Why minimal preprocessing for transformers:
  - BPE/SentencePiece tokenisers handle punctuation, casing, and rare words
    natively.  Aggressive preprocessing (stemming, stopword removal) destroys
    cues that the model relies on.
  - Example: "#Earthquake kills 50" → removing # and lowercasing loses the
    emphasis signal; removing "kills" as a stopword loses the key disaster cue.
  - Best practice: only remove noise that adds no semantic value (URLs,
    duplicate whitespace) and normalise high-variance tokens (mentions → @user).

Common mistakes in disaster tweet classification research:
  1. Using accuracy as the primary metric on imbalanced data.
  2. Aggressive text cleaning that removes hashtags and disaster keywords.
  3. Merging rare classes to reduce task difficulty (masks real-world hardness).
  4. Not performing stratified splits (random splits under-represent minorities).
  5. Evaluating on the dev set and reporting test set numbers from the best
     dev epoch — this leaks information; early-stop on dev, report final test.
  6. Not reporting per-class F1 — model may have zero recall on rare classes.
  7. Using fixed-length padding instead of dynamic padding (wastes GPU memory).
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig, TaskConfig

logger = logging.getLogger(__name__)


def build_model(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
) -> PreTrainedModel:
    """
    Instantiate a sequence-classification head on top of the pretrained encoder.
    id2label / label2id are embedded in the model config so that
    model.predict() and pipeline() work out of the box.
    """
    logger.info(
        f"Loading {model_cfg.key} ({model_cfg.hf_name}) "
        f"for task '{task_cfg.name}' ({task_cfg.num_classes} classes)"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_name,
        num_labels = task_cfg.num_classes,
        id2label   = task_cfg.id2label,
        label2id   = task_cfg.label2id,
        ignore_mismatched_sizes = True,   # classifier head is always reinit'd
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Parameters: {n_params:.1f}M")
    return model


def build_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the pretrained model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_name,
        use_fast = True,
    )
    return tokenizer


def model_param_count(model: PreTrainedModel) -> Dict[str, float]:
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return {"total_M": total, "trainable_M": trainable}
