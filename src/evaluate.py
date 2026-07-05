"""
Evaluation metrics for disaster tweet classification.

Metric choices and justifications:
  - Accuracy: reported for completeness but NOT used as the primary metric
    due to class imbalance (see loss.py for detailed reasoning).
  - Macro F1: primary metric.  Averages F1 per class with equal weight,
    penalising the model for ignoring minority classes.
  - Weighted F1: averages F1 weighted by support — useful secondary metric
    that reflects overall performance on the actual distribution.
  - Per-class F1: essential for identifying which classes the model struggles
    with.  A model with high macro F1 may still have near-zero F1 on the
    rarest class.
  - Confusion matrix: reveals systematic confusions (e.g. model conflates
    "injured_or_dead_people" with "affected_individuals").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Compute the full evaluation suite.

    Returns a dict suitable for JSON serialisation and comparison tables.
    """
    macro_f1    = f1_score(true_labels, pred_labels, average="macro",    zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_p     = precision_score(true_labels, pred_labels, average="macro",    zero_division=0)
    macro_r     = recall_score(   true_labels, pred_labels, average="macro",    zero_division=0)
    acc         = accuracy_score(true_labels, pred_labels)

    per_class_f1 = f1_score(
        true_labels, pred_labels,
        average=None, labels=list(range(len(label_names))),
        zero_division=0,
    )

    report = classification_report(
        true_labels, pred_labels,
        target_names = label_names,
        digits       = 4,
        zero_division = 0,
    )

    cm = confusion_matrix(true_labels, pred_labels).tolist()

    return {
        "accuracy":       round(float(acc),         4),
        "macro_f1":       round(float(macro_f1),     4),
        "weighted_f1":    round(float(weighted_f1),  4),
        "macro_precision":round(float(macro_p),      4),
        "macro_recall":   round(float(macro_r),      4),
        "per_class_f1":  {
            label_names[i]: round(float(per_class_f1[i]), 4)
            for i in range(len(label_names))
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_metrics(metrics: Dict, task_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Results — {task_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}  ← primary metric")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        bar = "█" * int(f1 * 20)
        print(f"    {cls:<50} {f1:.4f}  {bar}")
    print(f"\n{metrics['classification_report']}")


def save_metrics(metrics: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Remove classification_report string before JSON dump (already in txt)
    saveable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(path, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Metrics saved → {path}")


def save_report(metrics: Dict, path: str, header: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(header + "\n\n")
        f.write(metrics["classification_report"])
    logger.info(f"Classification report saved → {path}")
