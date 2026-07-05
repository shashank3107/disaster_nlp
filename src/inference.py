"""
Single-tweet and batch inference with saved model checkpoints.

Output format:
  {
    "text":          "original tweet text",
    "predicted":     "infrastructure_and_utility_damage",
    "confidence":    0.9123,
    "probabilities": {"class_a": 0.05, "class_b": 0.91, ...}
  }
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import preprocess_tweet

logger = logging.getLogger(__name__)


class DisasterInference:
    """
    Load a saved model and run inference on single tweets or batches.

    Args:
        model_dir: path to a directory saved via `trainer.save_pretrained()`
                   (must contain config.json, model.safetensors, tokenizer files)
        device:    'cuda' | 'cpu' | None (auto-detect)
        max_length: tokenisation max length (should match training)
    """

    def __init__(
        self,
        model_dir:  str,
        device:     Optional[str] = None,
        max_length: int = 128,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.max_length = max_length

        logger.info(f"Loading model from {model_dir} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).to(self.device)
        self.model.eval()

        # id2label is stored in model.config
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.label2id: Dict[str, int] = {
            v: int(k) for k, v in self.id2label.items()
        }
        logger.info(
            f"Model loaded: {len(self.id2label)} classes — {list(self.id2label.values())}"
        )

    @torch.no_grad()
    def predict_one(self, text: str, preprocess: bool = True) -> Dict:
        """
        Predict the class of a single tweet.

        Args:
            text:       raw tweet string
            preprocess: apply the same cleaning as training (recommended: True)

        Returns:
            dict with predicted, confidence, probabilities
        """
        if preprocess:
            text = preprocess_tweet(text)

        enc = self.tokenizer(
            text,
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        t0      = time.perf_counter()
        outputs = self.model(**enc)
        latency = (time.perf_counter() - t0) * 1000  # ms

        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu()
        pred_id = probs.argmax().item()

        return {
            "text":          text,
            "predicted":     self.id2label[pred_id],
            "confidence":    round(probs[pred_id].item(), 4),
            "probabilities": {
                self.id2label[i]: round(probs[i].item(), 4)
                for i in range(len(self.id2label))
            },
            "latency_ms":    round(latency, 2),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts:      List[str],
        batch_size: int = 32,
        preprocess: bool = True,
    ) -> List[Dict]:
        """
        Predict classes for a list of tweets.

        Processes in batches to avoid OOM on large inputs.
        Returns a list of dicts in the same format as predict_one().
        """
        if preprocess:
            texts = [preprocess_tweet(t) for t in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc   = self.tokenizer(
                chunk,
                padding        = True,
                truncation     = True,
                max_length     = self.max_length,
                return_tensors = "pt",
            )
            enc     = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            probs   = F.softmax(outputs.logits, dim=-1).cpu()

            for j, text in enumerate(chunk):
                pred_id = probs[j].argmax().item()
                results.append({
                    "text":          text,
                    "predicted":     self.id2label[pred_id],
                    "confidence":    round(probs[j][pred_id].item(), 4),
                    "probabilities": {
                        self.id2label[k]: round(probs[j][k].item(), 4)
                        for k in range(len(self.id2label))
                    },
                })
        return results

    def benchmark_speed(self, n_tweets: int = 100, batch_size: int = 32) -> Dict:
        """Measure average inference latency over n_tweets dummy inputs."""
        dummy = ["Flooding in city streets, rescue teams deployed #disaster"] * n_tweets
        t0    = time.perf_counter()
        _     = self.predict_batch(dummy, batch_size=batch_size, preprocess=False)
        total = time.perf_counter() - t0
        return {
            "n_tweets":           n_tweets,
            "total_sec":          round(total, 3),
            "tweets_per_sec":     round(n_tweets / total, 1),
            "avg_latency_ms":     round(total / n_tweets * 1000, 2),
        }
