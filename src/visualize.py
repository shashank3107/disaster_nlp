"""
Visualisation utilities: training curves, confusion matrices, comparison plots.
All plots use non-interactive Agg backend so they work headlessly on HPC.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Global style — clean, publication-ready
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize":13,
    "axes.labelsize":12,
    "figure.dpi":   150,
})


def plot_training_curves(
    history: Dict,          # keys: train_loss, val_loss, val_f1_macro
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot loss and F1 curves on twin axes."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=5)
    ax1.plot(epochs, history["val_loss"],   "b--s", label="Val Loss",  markersize=5)
    ax2.plot(epochs, history["val_f1_macro"], "r-^", label="Val Macro F1", markersize=5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss",      color="blue")
    ax2.set_ylabel("Macro F1",  color="red")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=9)

    plt.title(title)
    fig.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrix(
    cm: List[List[int]],
    label_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalise: bool = False,
) -> None:
    """Heatmap of the confusion matrix, optionally row-normalised."""
    cm_arr = np.array(cm)
    if normalise:
        row_sums = cm_arr.astype(float).sum(axis=1, keepdims=True)
        cm_arr   = np.divide(cm_arr.astype(float), row_sums, where=row_sums != 0)
        fmt      = ".2f"
    else:
        cm_arr = cm_arr.astype(int)
        fmt    = "d"

    short = [l.replace("_", "\n") for l in label_names]
    n     = len(label_names)
    fig_w = max(8, n * 1.5)
    fig_h = max(6, n * 1.3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm_arr,
        annot       = True,
        fmt         = fmt,
        xticklabels = short,
        yticklabels = short,
        cmap        = "Blues",
        linewidths  = 0.5,
        ax          = ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    _save(fig, save_path)


def plot_model_comparison(
    results: Dict[str, Dict],   # {model_key: metrics_dict}
    save_path: str,
    task_name: str = "",
) -> None:
    """
    Side-by-side bar chart comparing models on accuracy, macro F1,
    weighted F1.
    """
    models  = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels  = ["Accuracy", "Macro F1", "Weighted F1"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    x    = np.arange(len(models))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * w, vals, width=w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {task_name}")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    _save(fig, save_path)


def plot_per_class_f1_comparison(
    results: Dict[str, Dict],   # {model_key: metrics_dict with per_class_f1}
    label_names: List[str],
    save_path: str,
    task_name: str = "",
) -> None:
    """Grouped bar chart of per-class F1 for all models."""
    models = list(results.keys())
    n_cls  = len(label_names)
    n_mod  = len(models)
    x      = np.arange(n_cls)
    w      = 0.8 / n_mod
    palette = sns.color_palette("tab10", n_mod)

    fig, ax = plt.subplots(figsize=(max(12, n_cls * 1.8), 6))
    for i, (model, color) in enumerate(zip(models, palette)):
        pcf  = results[model].get("per_class_f1", {})
        vals = [pcf.get(l, 0) for l in label_names]
        ax.bar(x + i * w - (n_mod - 1) * w / 2, vals, width=w,
               label=model, color=color, alpha=0.8)

    short = [l.replace("_", "\n") for l in label_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 — {task_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save(fig, save_path)


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Plot saved → {path}")
