"""
SMMILe — Superpatch-based Measurable Multiple Instance Learning.

Full model for multiclass ovarian-cancer WSI classification (5 subtypes).

Architecture overview (two stages)
------------------------------------
Stage 1 (main.py --stage 1):
  frozen encoder  →  NICLayer (3×3 conv)  →  GatedAttention (per-category)
                  →  Instance Dropout (InD)  →  bag aggregation  →  bag logits

  Instance Sampling (InS): during training, the bag of K NIC features is first
  reduced to one representative per SLIC superpatch (stochastic sampling),
  creating a pseudo-bag h_s of size N_sp ≤ K.  Attention and aggregation are
  computed on h_s; the NIC features h (full K) are kept for Stage 2.

Stage 2 (main.py --stage 2):
  Stage 1 weights (frozen)  →  InstanceRefinement (N=3 linear layers)
  Each layer: v_n : R^{D_nic} → R^{C+1}  (C cancer classes + 1 background)
  Loss: L_cls + L_ref + L_mrf (see losses.py)

Paper equations referenced throughout
--------------------------------------
Eq. (4)–(6):   Instance Dropout (InD)
Eq. (7)–(9):   Delocalized Instance Sampling (InS) via SLIC superpatches
Eq. (10)–(13): Bag-level prediction from attention-weighted features
Eq. (14)–(16): Instance refinement with progressive pseudo-labels

References
----------
* Gao et al. (2025) — "SMMILe enables accurate spatial quantification in
  digital pathology using multiple-instance learning", Nature Cancer.
  DOI: 10.1038/s43018-025-01060-8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nic import NICLayer
from .attention import GatedAttention
from .instance_refinement import InstanceRefinement


class SMMILe(nn.Module):
    """Full SMMILe model for multiclass WSI classification.

    Parameters
    ----------
    embedding_dim:
        Pre-extracted encoder output dimensionality.
        512 for Conch (default), 1024 for ResNet-50 third-stage features.
    n_classes:
        Number of cancer subtypes C (5 for UBC-OCEAN ovarian task).
    nic_out_channels:
        Number of output channels from the NIC Conv2d layer (D_nic).
        256 per ``ovarian_conch_s1.yaml`` config.
    nic_kernel_size:
        Spatial kernel size for the NIC convolution.
        3 for the ovarian multiclass task; 1 for Camelyon16.
    attn_hidden_dim:
        Width L of the shared tanh/sigmoid projection inside GatedAttention.
        256 per config.
    attn_dropout:
        Whether to apply dropout inside GatedAttention.
    attn_dropout_rate:
        Dropout probability in GatedAttention.
    ind_drop_rate:
        Instance Dropout (InD) rate: fraction of the *top-scoring* instances
        to mask per category during training (Eq. 4–6 in SMMILe).
        0.0 disables InD.
    n_refinement_layers:
        Number of refinement layers N for Stage 2 (0 = Stage 1 only).
    ins_enabled:
        Enable Delocalized Instance Sampling (InS) during training.
        Should be True for Stage 1 training (Eq. 7–9).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_classes: int = 5,
        nic_out_channels: int = 256,
        nic_kernel_size: int = 3,
        attn_hidden_dim: int = 256,
        attn_dropout: bool = True,
        attn_dropout_rate: float = 0.25,
        ind_drop_rate: float = 0.5,
        n_refinement_layers: int = 0,
        ins_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ind_drop_rate = ind_drop_rate
        self.ins_enabled = ins_enabled

        # ----------------------------------------------------------------
        # Stage 1 components
        # ----------------------------------------------------------------

        # NIC: rearrange bag into 2-D grid, apply 3×3 conv (§3.1)
        self.nic = NICLayer(
            in_channels=embedding_dim,
            out_channels=nic_out_channels,
            kernel_size=nic_kernel_size,
        )

        # Per-category gated attention detector (§3.2, Eq. 10–11)
        self.attention = GatedAttention(
            in_dim=nic_out_channels,
            hidden_dim=attn_hidden_dim,
            n_classes=n_classes,
            dropout=attn_dropout,
            dropout_rate=attn_dropout_rate,
        )

        # Instance-level classifier shared across Stage 1 and Stage 2
        # Maps each instance feature h_k → (n_classes,) logit vector.
        # Used for both per-instance scoring and (via diagonal trick)
        # bag-level classification (Eq. 12–13).
        self.instance_classifier = nn.Linear(nic_out_channels, n_classes)

        # ----------------------------------------------------------------
        # Stage 2 component (optional)
        # ----------------------------------------------------------------

        # N=3 independent linear layers v_1..v_N : R^{D_nic} → R^{C+1}
        # Instantiated only when n_refinement_layers > 0.
        self.refinement: InstanceRefinement | None = None
        if n_refinement_layers > 0:
            self.refinement = InstanceRefinement(
                in_dim=nic_out_channels,
                n_classes=n_classes,
                n_layers=n_refinement_layers,
            )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _instance_dropout(
        self,
        A_raw: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Parameter-free Instance Dropout (InD) — SMMILe §3.2, Eq. (4)–(6).

        For each category c independently, the top-⌊ind_drop_rate × K⌋
        instances (those with the highest attention score a_{c,k}) are masked
        to zero, forcing the model to distribute attention more broadly.

        Applied only during training.  Per-category masking (not cross-
        category) is crucial so each category can independently suppress its
        own dominant instances without affecting other categories.

        Parameters
        ----------
        A_raw : (n_classes, K)
            Raw attention logits from GatedAttention (before softmax).
        A : (n_classes, K)
            Softmax-normalised attention weights (after GatedAttention).

        Returns
        -------
        A_drop : (n_classes, K)
            Attention weights after per-category dropout masking.
            The masked positions are set to −∞ in A_raw and the remaining
            positions are re-normalised via softmax to ensure valid weights.
        """
        if not self.training or self.ind_drop_rate <= 0.0:
            return A

        K = A.shape[1]
        n_drop = max(1, int(K * self.ind_drop_rate))

        # Mask raw logits with -inf at dropped positions, then re-softmax.
        # This preserves gradient flow through the kept instances.
        A_raw_drop = A_raw.clone()
        for c in range(self.n_classes):
            # Top-n_drop instances for category c (highest a_{c,k})
            top_idx = A[c].topk(n_drop).indices     # (n_drop,)
            A_raw_drop[c, top_idx] = float("-inf")  # effectively zero after softmax

        # Re-normalise: softmax over instances per category
        # Guard: if all positions are -inf (n_drop == K), return uniform.
        A_drop = F.softmax(A_raw_drop, dim=-1)

        # Replace NaN rows (all -inf → softmax gives NaN) with uniform
        nan_rows = A_drop.isnan().any(dim=-1)
        if nan_rows.any():
            A_drop[nan_rows] = 1.0 / K

        return A_drop

    def _instance_sampling(
        self,
        h: torch.Tensor,
        superpixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delocalized Instance Sampling (InS) — SMMILe §3.2, Eq. (7)–(9).

        Builds a pseudo-bag by randomly sampling **one** representative
        instance from each SLIC superpatch.  This creates a diverse,
        delocalized view of the bag at each training step, preventing the
        model from over-relying on the same discriminative instances.

        Parameters
        ----------
        h : (K, D_nic)
            Full-bag NIC feature matrix.
        superpixels : (K,)
            Superpatch ID for each instance (0-indexed, from SLIC).

        Returns
        -------
        h_pseudo : (N_sp, D_nic)
            Pseudo-bag features, one randomly chosen instance per superpatch.
        sp_ids : (N_sp,)
            Superpatch IDs corresponding to each row of h_pseudo
            (same order as ``torch.unique(superpixels)``).
        """
        sp_ids = superpixels.unique()  # sorted unique superpatch IDs
        sampled: list[torch.Tensor] = []
        for sp_id in sp_ids:
            idx_in_sp = (superpixels == sp_id).nonzero(as_tuple=True)[0]
            # Uniform random selection of one representative per superpatch
            rand_pos = torch.randint(
                len(idx_in_sp), (1,), device=h.device
            ).item()
            sampled.append(h[idx_in_sp[rand_pos]])  # (D_nic,)

        h_pseudo = torch.stack(sampled, dim=0)  # (N_sp, D_nic)
        return h_pseudo, sp_ids

    # ==================================================================
    # Forward pass
    # ==================================================================

    def forward(
        self,
        x: torch.Tensor,
        superpixels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Run SMMILe forward pass on one WSI bag.

        Parameters
        ----------
        x : (K, embedding_dim)
            Pre-extracted patch embeddings for all K tissue patches.
        superpixels : (K,) optional
            SLIC superpatch label per patch.  Required for InS during
            training and for L_mrf during Stage 2.

        Returns
        -------
        dict with keys:

        Stage 1 (always present)
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``bag_logits`` : (n_classes,)
            Bag-level classification logits (input to ClsLoss).
            Eq. (12)–(13): y_c = W_c^T M_c, where M_c = ∑_k a_{c,k} h_k.

        ``bag_prediction`` : (n_classes,)
            Alias for ``bag_logits`` (task-spec naming).

        ``attn_raw`` : (n_classes, K_bag)
            Raw attention logits (pre-softmax) from GatedAttention.
            K_bag = N_sp when InS is active (training), K otherwise.

        ``attn`` : (n_classes, K_bag)
            Softmax attention weights (input to RefLoss via Trainer).

        ``inst_features`` : (K, D_nic)
            Full-bag NIC features for ALL K instances (used in Stage 2
            refinement and for heatmap generation).

        ``inst_scores`` : (K, n_classes)
            Per-instance classification logits from ``instance_classifier``
            applied to the full-bag features (not attention-pooled).
            Eq. (10): f_c(h_k) for each instance k.

        ``attention_scores`` : (n_classes, K_bag)
            Alias for ``attn``.

        Stage 2 (present only when ``self.refinement is not None``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``ref_logits`` : (K, n_classes + 1)
            Raw logits from the **last** refinement layer v_N (used by
            Trainer → RefLoss, MRFLoss, and heatmap generation).
            Eq. (14): l_N(k) = v_N · h_k.

        ``ref_all_logits`` : list of N tensors, each (K, n_classes + 1)
            Raw logits from **all** N refinement layers (for progressive
            pseudo-label training in an extended Trainer).

        ``refinement_predictions`` : list of N tensors, each (K, n_classes + 1)
            Alias for ``ref_all_logits`` (task-spec naming).
        """
        # ----------------------------------------------------------
        # 1. NIC: (K, embedding_dim) → (K, D_nic)   (§3.1)
        # ----------------------------------------------------------
        h = self.nic(x)   # full-bag NIC features, used throughout

        # ----------------------------------------------------------
        # 2. Delocalized Instance Sampling (InS, §3.2, Eq. 7–9)
        #    Active during training when superpixels are provided.
        #    Produces a pseudo-bag h_bag from which attention is computed.
        # ----------------------------------------------------------
        if self.training and self.ins_enabled and superpixels is not None:
            h_bag, _ = self._instance_sampling(h, superpixels)  # (N_sp, D_nic)
        else:
            h_bag = h  # (K, D_nic)  — full bag at inference

        # ----------------------------------------------------------
        # 3. Gated attention on h_bag   (§3.2, Eq. 10–11)
        #    A_raw, A : (n_classes, K_bag)
        # ----------------------------------------------------------
        A_raw, A = self.attention(h_bag)

        # ----------------------------------------------------------
        # 4. Instance Dropout (InD, §3.2, Eq. 4–6)
        #    Per-category masking of top-scoring instances.
        # ----------------------------------------------------------
        A_drop = self._instance_dropout(A_raw, A)  # (n_classes, K_bag)

        # ----------------------------------------------------------
        # 5. Bag aggregation (§3.2, Eq. 12)
        #    M_c = ∑_k a_{c,k} h_k   →   M : (n_classes, D_nic)
        # ----------------------------------------------------------
        M = torch.mm(A_drop, h_bag)  # (n_classes, D_nic)

        # ----------------------------------------------------------
        # 6. Bag-level logits (§3.2, Eq. 13)
        #    y_c = W_c^T M_c   — the diagonal trick:
        #      instance_classifier: Linear(D_nic, n_classes)
        #      instance_classifier(M) : (n_classes, n_classes)
        #                              where [i,j] = W[j] · M[i]
        #      diagonal [c,c] = W[c] · M[c]  ✓  (c-th classifier × c-th bag repr.)
        # ----------------------------------------------------------
        bag_logits = self.instance_classifier(M).diagonal()  # (n_classes,)

        # Per-instance classification scores from the same linear layer
        # applied to ALL K instances (not attention-pooled).
        # Used for spatial heatmap generation and evaluation.
        inst_scores = self.instance_classifier(h)  # (K, n_classes)

        out: dict = {
            # Primary outputs (trainer-compatible naming)
            "bag_logits":        bag_logits,           # (n_classes,)
            "attn_raw":          A_raw,                # (n_classes, K_bag)
            "attn":              A,                    # (n_classes, K_bag) — post-InD re-norm
            "inst_features":     h,                    # (K, D_nic)
            "inst_scores":       inst_scores,          # (K, n_classes)
            # Task-spec aliases
            "bag_prediction":    bag_logits,
            "attention_scores":  A,
        }

        # ----------------------------------------------------------
        # 7. Stage 2 — instance refinement (§3.3, Eq. 14–16)
        #    Applied to full-bag features h (all K instances).
        # ----------------------------------------------------------
        if self.refinement is not None:
            ref_all: list[torch.Tensor] = self.refinement(h)  # N × (K, C+1)
            out["ref_logits"]              = ref_all[-1]  # last layer (for Trainer)
            out["ref_all_logits"]          = ref_all      # all layers
            out["refinement_predictions"]  = ref_all      # task-spec alias

        return out
