"""
Evaluation metrics for SMMILe.

WSI-level:   macro AUC (one-vs-rest)
Patch-level: macro AUC, macro F1, accuracy, precision, recall
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    wsi_auc: float = 0.0
    patch_auc: float = 0.0
    patch_f1: float = 0.0
    patch_acc: float = 0.0
    patch_precision: float = 0.0
    patch_recall: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"WSI AUC={self.wsi_auc:.4f} | "
            f"Patch AUC={self.patch_auc:.4f} F1={self.patch_f1:.4f} "
            f"Acc={self.patch_acc:.4f}"
        )


class Evaluator:
    """Accumulates predictions across a dataset split and computes metrics.

    Usage::

        ev = Evaluator(n_classes=5)
        for batch in dataloader:
            ev.update_wsi(slide_id, bag_probs, true_label)
            ev.update_patch(patch_probs, patch_true_labels)
        result = ev.compute()
    """

    def __init__(self, n_classes: int = 5) -> None:
        self.n_classes = n_classes
        self._wsi_probs: list[np.ndarray] = []
        self._wsi_labels: list[int] = []
        self._patch_probs: list[np.ndarray] = []
        self._patch_labels: list[int] = []

    def reset(self) -> None:
        self._wsi_probs.clear()
        self._wsi_labels.clear()
        self._patch_probs.clear()
        self._patch_labels.clear()

    def update_wsi(
        self,
        slide_id: str,
        bag_probs: np.ndarray,
        true_label: int,
    ) -> None:
        """Record WSI-level prediction.

        Args:
            slide_id:   Identifier (for logging).
            bag_probs:  Softmax probabilities (n_classes,).
            true_label: Ground-truth class index.
        """
        self._wsi_probs.append(bag_probs)
        self._wsi_labels.append(true_label)

    def update_patch(
        self,
        patch_probs: np.ndarray,
        patch_labels: np.ndarray,
    ) -> None:
        """Record patch-level predictions for a single WSI.

        Args:
            patch_probs:  (N, n_classes) softmax probabilities.
            patch_labels: (N,) ground-truth class indices.
        """
        self._patch_probs.append(patch_probs)
        self._patch_labels.extend(patch_labels.tolist())

    def compute(self) -> EvalResult:
        result = EvalResult()

        # ---- WSI-level AUC ----
        if len(self._wsi_labels) > 0:
            wsi_prob_arr = np.stack(self._wsi_probs, axis=0)  # (M, C)
            wsi_label_arr = np.array(self._wsi_labels)
            try:
                result.wsi_auc = roc_auc_score(
                    wsi_label_arr,
                    wsi_prob_arr,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError as e:
                logger.warning("Could not compute WSI AUC: %s", e)

        # ---- Patch-level metrics ----
        if len(self._patch_labels) > 0:
            patch_prob_arr = np.concatenate(self._patch_probs, axis=0)  # (P, C)
            patch_label_arr = np.array(self._patch_labels)
            patch_pred_arr = patch_prob_arr.argmax(axis=1)

            try:
                result.patch_auc = roc_auc_score(
                    patch_label_arr,
                    patch_prob_arr,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError as e:
                logger.warning("Could not compute patch AUC: %s", e)

            result.patch_acc = accuracy_score(patch_label_arr, patch_pred_arr)
            result.patch_f1 = f1_score(
                patch_label_arr, patch_pred_arr, average="macro", zero_division=0
            )
            result.patch_precision = precision_score(
                patch_label_arr, patch_pred_arr, average="macro", zero_division=0
            )
            result.patch_recall = recall_score(
                patch_label_arr, patch_pred_arr, average="macro", zero_division=0
            )

        return result


# ---------------------------------------------------------------------------
# Fold aggregation utility
# ---------------------------------------------------------------------------

def summarize_folds(fold_results: list[EvalResult]) -> dict[str, str]:
    """Compute mean ± std across folds.

    Args:
        fold_results: List of EvalResult, one per fold.

    Returns:
        Dict mapping metric name → "mean±std" string.
    """
    metrics = ["wsi_auc", "patch_auc", "patch_f1", "patch_acc", "patch_precision", "patch_recall"]
    summary: dict[str, str] = {}
    for m in metrics:
        vals = np.array([getattr(r, m) for r in fold_results])
        summary[m] = f"{vals.mean():.4f}±{vals.std():.4f}"
    return summary
