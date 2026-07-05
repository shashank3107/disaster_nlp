"""
Centralised configuration for all tasks and models.

Design choices:
  - Dataclasses for static typing and easy serialisation.
  - Tasks and models are fully independent: any model × any task is valid.
  - Hyperparameters follow well-established defaults for transformer fine-tuning
    on small social-media corpora (Devlin et al., 2019; He et al., 2021).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json, os


# ─────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    name: str
    train_file: str
    dev_file: str
    test_file: str
    label_col: str                   # exact column name in the TSV
    labels: List[str]                # canonical ordered label list
    description: str

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def label2id(self) -> Dict[str, int]:
        return {l: i for i, l in enumerate(self.labels)}

    @property
    def id2label(self) -> Dict[int, str]:
        return {i: l for i, l in enumerate(self.labels)}


TASKS: Dict[str, TaskConfig] = {
    "informative": TaskConfig(
        name        = "informative",
        train_file  = "task_informative_text_img_train.tsv",
        dev_file    = "task_informative_text_img_dev.tsv",
        test_file   = "task_informative_text_img_test.tsv",
        label_col   = "label_text",
        labels      = ["informative", "not_informative"],
        description = "Binary: Is the tweet disaster-informative?",
    ),
    "humanitarian": TaskConfig(
        name        = "humanitarian",
        train_file  = "task_humanitarian_text_img_train.tsv",
        dev_file    = "task_humanitarian_text_img_dev.tsv",
        test_file   = "task_humanitarian_text_img_test.tsv",
        label_col   = "label_text",
        labels      = [
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "injured_or_dead_people",
            "missing_or_found_people",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
            "vehicle_damage",
        ],
        description = "8-class: Humanitarian category of tweet",
    ),
    "damage": TaskConfig(
        name        = "damage",
        train_file  = "task_damage_text_img_train.tsv",
        dev_file    = "task_damage_text_img_dev.tsv",
        test_file   = "task_damage_text_img_test.tsv",
        label_col   = "label",
        labels      = ["little_or_no_damage", "mild_damage", "severe_damage"],
        description = "3-class: Physical damage severity",
    ),
}


# ─────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str                          # short identifier used in paths
    hf_name: str                      # HuggingFace model hub name
    max_length: int = 128
    description: str = ""

    # DeBERTa uses token_type_ids=False by default in its tokenizer
    # but we let HF handle that automatically.


MODELS: Dict[str, ModelConfig] = {
    # ── Base models ───────────────────────────────────────────
    "deberta": ModelConfig(
        key         = "deberta",
        hf_name     = "microsoft/deberta-v3-base",
        max_length  = 128,
        description = "DeBERTa-v3-base: disentangled attention + enhanced mask decoder",
    ),
    "roberta": ModelConfig(
        key         = "roberta",
        hf_name     = "roberta-base",
        max_length  = 128,
        description = "RoBERTa-base: robustly optimised BERT pretraining",
    ),
    "distilbert": ModelConfig(
        key         = "distilbert",
        hf_name     = "distilbert-base-uncased",
        max_length  = 128,
        description = "DistilBERT: lightweight 66M-param BERT distillation (baseline)",
    ),

    # ── Large models ──────────────────────────────────────────
    # DeBERTa-v3-large: ~400M params, best in class for NLU tasks
    "deberta-large": ModelConfig(
        key         = "deberta-large",
        hf_name     = "microsoft/deberta-v3-large",
        max_length  = 128,
        description = "DeBERTa-v3-large: ~400M params, SOTA NLU encoder",
    ),
    # RoBERTa-large: ~355M params
    "roberta-large": ModelConfig(
        key         = "roberta-large",
        hf_name     = "roberta-large",
        max_length  = 128,
        description = "RoBERTa-large: ~355M params, robustly optimised large BERT",
    ),
    # DistilBERT has no official large variant — BERT-large is the natural
    # substitute: same architecture at 2× depth (24 layers vs 12).
    "bert-large": ModelConfig(
        key         = "bert-large",
        hf_name     = "bert-large-uncased",
        max_length  = 128,
        description = "BERT-large-uncased: ~340M params, large baseline (no DistilBERT-large exists)",
    ),
}


# ─────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    data_dir:    str = "./data"
    output_dir:  str = "./experiments"
    log_dir:     str = "./logs"

    # Core hyperparams — chosen based on standard transformer fine-tuning
    # practice: LR 1e-5 to 3e-5, batch 16-32, 3-5 epochs.
    learning_rate:       float = 2e-5
    weight_decay:        float = 0.01
    warmup_ratio:        float = 0.10
    num_epochs:          int   = 5
    batch_size:          int   = 16        # per-device
    eval_batch_size:     int   = 32
    grad_accum_steps:    int   = 2         # effective batch = 32
    max_grad_norm:       float = 1.0
    early_stopping_patience: int = 3

    # AMP
    fp16: bool = True                      # set False on CPU or older GPUs

    # DataLoader
    num_workers: int = 4
    pin_memory:  bool = True

    # Loss
    loss_type: str = "weighted_ce"         # "weighted_ce" | "focal"
    focal_gamma: float = 2.0
    max_class_weight: float = 10.0        # cap to prevent instability

    # Checkpointing
    save_best_only: bool = True
    metric_for_best: str = "macro_f1"

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
