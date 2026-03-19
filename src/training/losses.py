"""
Loss functions for SMMILe training.

L_cls — Bag-level BCE classification loss (Stage 1 + Stage 2)
L_ref — Instance refinement cross-entropy loss (Stage 2)
L_mrf — Superpatch-based MRF smoothness loss (Stage 2)
L_cons — Normal-class consistency loss (Stage 2; NOT used for UBC-OCEAN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L_cls — Bag classification loss (BCE per category)
# ---------------------------------------------------------------------------

class ClsLoss(nn.Module):
    """Binary cross-entropy loss computed independently per category.

    For multiclass tasks (UBC-OCEAN) each category is treated as a one-vs-rest
    binary problem at the bag level, following the SMMILe formulation.

    Args:
        n_classes: Number of cancer subtypes.
    """

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        bag_logits: torch.Tensor,
        label: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bag_logits: (n_classes,) raw logits from the model.
            label:      Scalar int (0–n_classes-1) or (n_classes,) one-hot.

        Returns:
            Scalar loss.
        """
        if isinstance(label, int):
            target = torch.zeros(self.n_classes, device=bag_logits.device)
            target[label] = 1.0
        else:
            target = label.float()
        return self.bce(bag_logits, target)


# ---------------------------------------------------------------------------
# L_ref — Instance refinement loss (pseudo-label cross-entropy)
# ---------------------------------------------------------------------------

class RefLoss(nn.Module):
    """Cross-entropy loss on top-θ% pseudo-labeled instances.

    For each category, the top-θ% highest-attention instances are selected
    as pseudo-positive labels and the bottom-θ% as pseudo-negative (background).

    Args:
        theta:     Fraction of instances to pseudo-label (default 0.10 = 10%).
        n_classes: Number of cancer subtypes (C). Output logits are C+1-dim.
    """

    def __init__(self, theta: float = 0.10, n_classes: int = 5) -> None:
        super().__init__()
        self.theta = theta
        self.n_classes = n_classes

    def forward(
        self,
        ref_logits: torch.Tensor,
        attn: torch.Tensor,
        label: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_logits: Instance logits (N, C+1).
            attn:       Attention weights (n_classes, N) from Stage 1 detector.
            label:      Ground-truth bag label (int or (n_classes,) one-hot).

        Returns:
            Scalar loss.
        """
        N = ref_logits.shape[0]
        n_select = max(1, int(N * self.theta))

        if isinstance(label, int):
            pos_class = label
        else:
            pos_class = int(label.argmax().item())

        # Top-θ% instances for positive class → pseudo-label = pos_class
        top_idx = attn[pos_class].topk(n_select).indices
        # Bottom-θ% instances for positive class → pseudo-label = C (background)
        bot_idx = attn[pos_class].topk(n_select, largest=False).indices

        sel_idx = torch.cat([top_idx, bot_idx])
        pseudo_labels = torch.cat([
            torch.full((n_select,), pos_class, dtype=torch.long, device=ref_logits.device),
            torch.full((n_select,), self.n_classes, dtype=torch.long, device=ref_logits.device),
        ])

        loss = F.cross_entropy(ref_logits[sel_idx], pseudo_labels)
        return loss


# ---------------------------------------------------------------------------
# L_mrf — Superpatch MRF smoothness loss
# ---------------------------------------------------------------------------

class MRFLoss(nn.Module):
    """Superpatch-based MRF smoothness loss.

    Encourages spatially adjacent instances (sharing a superpixel) to have
    similar predicted class distributions.

    Loss = λ1 * (intra-superpatch variance) + λ2 * (cross-entropy unary term)

    Args:
        lambda1: Weight for the spatial smoothness term.
        lambda2: Weight for the unary (classification confidence) term.
        n_classes: Number of cancer subtypes.
    """

    def __init__(
        self,
        lambda1: float = 0.8,
        lambda2: float = 0.2,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_classes = n_classes

    def forward(
        self,
        ref_logits: torch.Tensor,
        superpixels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_logits:  Instance logits (N, C+1).
            superpixels: Superpixel label per instance (N,).

        Returns:
            Scalar MRF loss.
        """
        probs = F.softmax(ref_logits[:, : self.n_classes], dim=-1)  # (N, C)
        smoothness_loss = torch.tensor(0.0, device=ref_logits.device)

        sp_ids = superpixels.unique()
        for sp_id in sp_ids:
            mask = superpixels == sp_id
            if mask.sum() < 2:
                continue
            sp_probs = probs[mask]  # (K, C)
            # Variance across patches within the superpatch
            sp_var = sp_probs.var(dim=0).mean()
            smoothness_loss = smoothness_loss + sp_var

        n_sp = max(1, len(sp_ids))
        smoothness_loss = smoothness_loss / n_sp

        # Unary: entropy of predicted distribution (encourage confident predictions)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        unary_loss = entropy

        return self.lambda1 * smoothness_loss + self.lambda2 * unary_loss


# ---------------------------------------------------------------------------
# L_cons — Normal-class consistency loss (NOT used for UBC-OCEAN)
# ---------------------------------------------------------------------------

class ConsLoss(nn.Module):
    """Consistency loss for datasets with an explicit normal class.

    Not used for UBC-OCEAN (all WSIs are cancerous).
    Included for completeness / future datasets.

    Args:
        normal_class_idx: Index of the normal/background class in the label set.
    """

    def __init__(self, normal_class_idx: int = -1) -> None:
        super().__init__()
        self.normal_class_idx = normal_class_idx

    def forward(
        self,
        ref_logits: torch.Tensor,
        label: int | torch.Tensor,
    ) -> torch.Tensor:
        """Returns zero loss for non-normal bags; entropy penalty for normal bags."""
        if isinstance(label, int):
            is_normal = label == self.normal_class_idx
        else:
            is_normal = bool(label[self.normal_class_idx].item() > 0.5)

        if not is_normal:
            return torch.tensor(0.0, device=ref_logits.device)

        # For normal bags: encourage all instances to predict background
        probs = torch.softmax(ref_logits, dim=-1)
        bg_prob = probs[:, -1]  # last column = background
        loss = -bg_prob.log().mean()
        return loss
