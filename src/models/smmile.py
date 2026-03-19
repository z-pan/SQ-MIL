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
            Main bag-level logits: InS+InD at train, full-bag/drop at eval.
            Eq. (12)–(13): y_c = W_c^T M_c, where M_c = ∑_k a_{c,k} h_k.
        ``bag_logits_raw`` : (n_classes,)
            Variant 1 — full bag, no InD.
        ``bag_logits_drop`` : (n_classes,)
            Variant 2 — full bag, with per-category InD.
        ``bag_logits_sampled`` : (n_classes,)
            Variant 3 — InS pseudo-bag, no InD (= raw at eval).
        ``bag_prediction`` : (n_classes,)
            Alias for ``bag_logits``.
        ``attn`` : (n_classes, K)
            Full-bag softmax attention weights.  **Always (C, K)** — never
            the pseudo-bag shape — so RefLoss can safely index ref_logits.
        ``attn_raw`` : (n_classes, K)
            Full-bag raw attention logits (pre-softmax).
        ``inst_features`` : (K, D_nic)
            Full-bag NIC features for all K instances.
        ``inst_scores`` : (K, n_classes)
            Per-instance classification logits (not attention-pooled).
            Used for heatmap generation and spatial evaluation.
        ``attention_scores`` : (n_classes, K)
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
        # 2. Full-bag attention — ALWAYS computed over all K instances.
        #    This is the canonical attention used by:
        #      • RefLoss (pseudo-label selection over K instances)
        #      • ConsistencyLoss (uniform attention for normal bags)
        #      • Heatmap generation (attention weights per patch)
        #    A_raw_full, A_full : (n_classes, K)
        # ----------------------------------------------------------
        A_raw_full, A_full = self.attention(h)  # (C, K)

        # ----------------------------------------------------------
        # 3. Four bag-prediction variants (Eq. 14 in SMMILe).
        #    The ClassificationLoss averages BCE over all four to
        #    encourage robust predictions under different regularisations.
        #
        #    variant 1 — raw:              full bag, no InD
        #    variant 2 — drop:             full bag, with InD
        #    variant 3 — sampled:          InS pseudo-bag, no InD
        #    variant 4 — drop+sampled:     InS pseudo-bag, with InD  ← bag_logits
        #
        #    At eval / when InS is off, variants 3 and 4 collapse to
        #    variants 1 and 2 respectively (h_bag = h, no stochasticity).
        # ----------------------------------------------------------

        # --- Variant 1: raw (full bag, no InD)
        M_raw = torch.mm(A_full, h)                           # (C, D_nic)
        bag_logits_raw = self.instance_classifier(M_raw).diagonal()  # (C,)

        # --- Variant 2: drop (full bag, with InD)
        A_drop_full = self._instance_dropout(A_raw_full, A_full)  # (C, K)
        M_drop = torch.mm(A_drop_full, h)
        bag_logits_drop = self.instance_classifier(M_drop).diagonal()  # (C,)

        # --- Variants 3 & 4: InS pseudo-bag (training only)
        if self.training and self.ins_enabled and superpixels is not None:
            h_bag, _ = self._instance_sampling(h, superpixels)   # (N_sp, D_nic)
            A_raw_bag, A_bag = self.attention(h_bag)              # (C, N_sp)

            # Variant 3: sampled, no InD
            M_sampled = torch.mm(A_bag, h_bag)
            bag_logits_sampled = self.instance_classifier(M_sampled).diagonal()

            # Variant 4: sampled + InD  ← primary prediction at train time
            A_drop_bag = self._instance_dropout(A_raw_bag, A_bag)
            M_drop_bag = torch.mm(A_drop_bag, h_bag)
            bag_logits = self.instance_classifier(M_drop_bag).diagonal()
        else:
            # At eval / InS disabled: variants 3 & 4 = variants 1 & 2
            bag_logits_sampled = bag_logits_raw
            bag_logits         = bag_logits_drop

        # ----------------------------------------------------------
        # 4. Per-instance classification logits (for heatmaps & eval)
        # ----------------------------------------------------------
        inst_scores = self.instance_classifier(h)  # (K, n_classes)

        out: dict = {
            # ---- Four bag prediction variants for ClassificationLoss ----
            "bag_logits":          bag_logits,          # (C,) main (InS+InD / drop at eval)
            "bag_logits_raw":      bag_logits_raw,      # (C,) full bag, no InD
            "bag_logits_drop":     bag_logits_drop,     # (C,) full bag, with InD
            "bag_logits_sampled":  bag_logits_sampled,  # (C,) pseudo-bag, no InD
            # ---- Full-bag attention — ALWAYS (C, K) ----
            # Invariant: out["attn"] is always over the full K instances so that
            # RefLoss can correctly index ref_logits[K, C+1] by patch position.
            "attn":                A_full,              # (C, K) normalised
            "attn_raw":            A_raw_full,          # (C, K) raw logits
            # ---- Features ----
            "inst_features":       h,                   # (K, D_nic)
            "inst_scores":         inst_scores,         # (K, C)
            # ---- Aliases (task-spec naming) ----
            "bag_prediction":      bag_logits,
            "attention_scores":    A_full,
        }

        # ----------------------------------------------------------
        # 5. Stage 2 — instance refinement (§3.3, Eq. 14–16)
        #    Applied to full-bag features h (all K instances).
        # ----------------------------------------------------------
        if self.refinement is not None:
            ref_all: list[torch.Tensor] = self.refinement(h)  # N × (K, C+1)
            out["ref_logits"]             = ref_all[-1]   # last layer (Trainer compat)
            out["ref_all_logits"]         = ref_all       # all N layers
            out["refinement_predictions"] = ref_all       # task-spec alias

        return out
