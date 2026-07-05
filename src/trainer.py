"""
Custom PyTorch training loop.

Design decisions:
  - Pure PyTorch (no HuggingFace Trainer) for full control over AMP, gradient
    accumulation, and checkpoint logic.  This is important for HPC jobs where
    you may need to resume from a specific step.
  - AMP (torch.cuda.amp) with GradScaler for FP16: reduces GPU memory by ~40%
    and speeds up training on Ampere/Volta GPUs with no accuracy loss.
  - Gradient accumulation: allows effective batch sizes of 32–64 even when
    physical batch is 16 (limited by GPU VRAM).
  - Gradient clipping (max_norm=1.0): stabilises training, especially for
    DeBERTa which can produce large gradients early in fine-tuning.
  - AdamW + linear warmup + linear decay: standard recipe for transformer
    fine-tuning.  Warmup prevents the classifier head from dominating early.
  - Early stopping on val Macro F1: halts training before overfitting; patience
    of 3 means the model tolerates 3 non-improving epochs.
  - Checkpoint saving: best model saved to disk; training can resume from any
    checkpoint (see `resume_from_checkpoint`).
"""

from __future__ import annotations

import json
import logging
import os
import time

# Suppress HuggingFace tokenizers parallelism warning when forking DataLoader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from .config import TrainConfig
from .evaluate import compute_all_metrics, print_metrics, save_metrics, save_report
from .visualize import plot_training_curves

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GPU utilities
# ─────────────────────────────────────────────────────────────

def gpu_memory_gb() -> str:
    if not torch.cuda.is_available():
        return "N/A (CPU)"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    return f"{allocated:.2f}GB alloc / {reserved:.2f}GB reserved"


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class DisasterTrainer:
    """
    Self-contained trainer: train, evaluate, save, resume.

    Args:
        model:       pretrained model with classification head
        tokenizer:   matching tokenizer (for collator and saving)
        loss_fn:     loss module (WeightedCE or FocalLoss)
        train_ds:    PyTorch Dataset for training
        val_ds:      PyTorch Dataset for validation
        cfg:         TrainConfig dataclass
        output_dir:  root directory for checkpoints and logs
        label_names: ordered list of class names (for metrics)
    """

    def __init__(
        self,
        model:       PreTrainedModel,
        tokenizer:   PreTrainedTokenizerBase,
        loss_fn:     nn.Module,
        train_ds:    Dataset,
        val_ds:      Dataset,
        cfg:         TrainConfig,
        output_dir:  str,
        label_names: List[str],
    ) -> None:
        self.model       = model
        self.tokenizer   = tokenizer
        self.loss_fn     = loss_fn
        self.train_ds    = train_ds
        self.val_ds      = val_ds
        self.cfg         = cfg
        self.output_dir  = Path(output_dir)
        self.label_names = label_names
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.output_dir / "best"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.best_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss":    [],
            "val_loss":      [],
            "val_f1_macro":  [],
            "val_accuracy":  [],
        }
        self.best_score   = -1.0
        self.best_epoch   = 0
        self.no_improve   = 0

    # ── DataLoaders ───────────────────────────────────────────

    def _make_loaders(self) -> Tuple[DataLoader, DataLoader]:
        collator = DataCollatorWithPadding(
            tokenizer   = self.tokenizer,
            pad_to_multiple_of = 8 if self.cfg.fp16 else None,
        )
        # num_workers=0 on CPU avoids tokenizers fork/parallelism conflicts
        nw = self.cfg.num_workers if self.device.type == "cuda" else 0
        train_loader = DataLoader(
            self.train_ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = True,
            num_workers = nw,
            pin_memory  = self.cfg.pin_memory and self.device.type == "cuda",
            collate_fn  = collator,
            drop_last   = False,
        )
        val_loader = DataLoader(
            self.val_ds,
            batch_size  = self.cfg.eval_batch_size,
            shuffle     = False,
            num_workers = nw,
            pin_memory  = self.cfg.pin_memory and self.device.type == "cuda",
            collate_fn  = collator,
        )
        return train_loader, val_loader

    # ── Training ──────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> Dict:
        """
        Full training loop.

        Returns:
            dict with best_epoch, best_score, history, and final metrics.
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        train_loader, val_loader = self._make_loaders()

        # Optimizer
        no_decay    = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=self.cfg.learning_rate)

        total_steps   = (len(train_loader) // self.cfg.grad_accum_steps) * self.cfg.num_epochs
        warmup_steps  = int(total_steps * self.cfg.warmup_ratio)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        scaler = GradScaler("cuda", enabled=self.cfg.fp16 and self.device.type == "cuda")

        start_epoch = 1
        if resume_from:
            start_epoch = self._load_checkpoint(
                resume_from, optimizer, scheduler, scaler
            ) + 1
            logger.info(f"Resumed from checkpoint, starting epoch {start_epoch}")

        logger.info(
            f"\nTraining on {self.device}  |  "
            f"epochs={self.cfg.num_epochs}  |  "
            f"batch={self.cfg.batch_size}×{self.cfg.grad_accum_steps}={self.cfg.batch_size*self.cfg.grad_accum_steps}  |  "
            f"steps={total_steps}  |  warmup={warmup_steps}"
        )

        for epoch in range(start_epoch, self.cfg.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(
                epoch, train_loader, optimizer, scheduler, scaler
            )
            val_metrics = self._evaluate(val_loader)
            val_loss    = self._val_loss(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1_macro"].append(val_metrics["macro_f1"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            elapsed = time.time() - t0
            logger.info(
                f"\nEpoch {epoch}/{self.cfg.num_epochs}  "
                f"| train_loss={train_loss:.4f}  "
                f"| val_loss={val_loss:.4f}  "
                f"| macro_f1={val_metrics['macro_f1']:.4f}  "
                f"| acc={val_metrics['accuracy']:.4f}  "
                f"| {elapsed/60:.1f}min  "
                f"| GPU: {gpu_memory_gb()}"
            )

            # Checkpoint
            self._save_checkpoint(epoch, optimizer, scheduler, scaler)

            # Best model
            score = val_metrics[self.cfg.metric_for_best]
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.no_improve = 0
                self._save_best()
                logger.info(
                    f"  ✓ New best {self.cfg.metric_for_best}={score:.4f} → saved"
                )
            else:
                self.no_improve += 1
                logger.info(
                    f"  No improvement ({self.no_improve}/{self.cfg.early_stopping_patience})"
                )

            if self.no_improve >= self.cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # Save training curves
        plot_training_curves(
            self.history,
            save_path = str(self.output_dir / "training_curves.png"),
            title     = f"Training — {self.output_dir.name}",
        )
        self._save_history()

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "history":    self.history,
        }

    def _train_epoch(
        self,
        epoch:        int,
        loader:       DataLoader,
        optimizer:    AdamW,
        scheduler,
        scaler:       GradScaler,
    ) -> float:
        self.model.train()
        total_loss     = 0.0
        steps_done     = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, ncols=100)
        for step, batch in enumerate(pbar):
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with autocast("cuda", enabled=self.cfg.fp16 and self.device.type == "cuda"):
                outputs = self.model(**batch)
                loss    = self.loss_fn(outputs.logits, labels)
                loss    = loss / self.cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                steps_done += 1

            total_loss += loss.item() * self.cfg.grad_accum_steps
            pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(loader, desc="  [eval]", leave=False, ncols=100):
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with autocast("cuda", enabled=self.cfg.fp16 and self.device.type == "cuda"):
                outputs = self.model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        return compute_all_metrics(
            np.array(all_labels), np.array(all_preds), self.label_names
        )

    @torch.no_grad()
    def _val_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for batch in loader:
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with autocast("cuda", enabled=self.cfg.fp16 and self.device.type == "cuda"):
                outputs = self.model(**batch)
                loss    = self.loss_fn(outputs.logits, labels)
            total += loss.item()
        return total / len(loader)

    # ── Evaluate on test set ──────────────────────────────────

    def evaluate_test(
        self, test_ds: Dataset, save_dir: Optional[str] = None
    ) -> Dict:
        """Run inference on test set; optionally save metrics & report."""
        collator    = DataCollatorWithPadding(
            tokenizer   = self.tokenizer,
            pad_to_multiple_of = 8 if self.cfg.fp16 else None,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = self.cfg.eval_batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = self.cfg.pin_memory and self.device.type == "cuda",
            collate_fn  = collator,
        )
        metrics = self._evaluate(test_loader)

        if save_dir:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            save_metrics(metrics, str(sd / "test_metrics.json"))
            save_report(metrics, str(sd / "test_report.txt"))

        return metrics

    # ── Checkpoint I/O ────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, optimizer, scheduler, scaler) -> None:
        path = self.ckpt_dir / f"epoch_{epoch:03d}"
        path.mkdir(exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        torch.save({
            "epoch":     epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "history":   self.history,
            "best_score":self.best_score,
        }, str(path / "trainer_state.pt"))

    def _load_checkpoint(self, checkpoint_dir: str, optimizer, scheduler, scaler) -> int:
        state = torch.load(
            str(Path(checkpoint_dir) / "trainer_state.pt"),
            map_location=self.device,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        self.history    = state["history"]
        self.best_score = state["best_score"]
        return state["epoch"]

    def _save_best(self) -> None:
        self.model.save_pretrained(str(self.best_dir))
        self.tokenizer.save_pretrained(str(self.best_dir))

    def _save_history(self) -> None:
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
