"""
Instance refinement network for SMMILe Stage 2.

Architecture
------------
N = 3 independent linear layers v_1, v_2, v_3, each mapping the NIC instance
features h_k ∈ R^{D_nic} directly to (C+1)-dimensional logits:

    l_n(k) = v_n · h_k + b_n,   v_n ∈ R^{(C+1)×D_nic},   n = 1, 2, 3

where C is the number of cancer subtypes and the (C+1)-th output is the
background / non-informative class.  All N layers see the *same* input h_k;
the "progressive" nature of refinement is realised during **training** by
using layer n-1's softmax outputs to generate pseudo-labels that supervise
layer n (handled in the Trainer, not in this module).

The forward pass returns raw logits for all N layers so the Trainer can
compute the RefLoss on each layer independently (or cumulatively).

Pseudo-label generation
-----------------------
``select_pseudo_labels`` (static method) implements the top-θ% / bottom-θ%
selection strategy described in SMMILe §3.3:

    For each class c ∈ {0 .. C-1}:
      • The instances with the highest score[:, c] in the top θ fraction
        are assigned pseudo-label c (positive).
      • Instances with the lowest max-class probability (bottom θ fraction
        of the *unassigned* instances) are assigned pseudo-label C
        (background / negative).
      • All other instances get label -1 (ignored in the CE loss).

    Conflicts (an instance selected as positive for two classes) are resolved
    by keeping the class with the higher confidence score.

Paper equations (SMMILe §3.3, progressive pseudo-label supervision)
-------------------------------------------------------------------
  Eq. (14): instance-level prediction at refinement layer n:
      p_n(k) = softmax( v_n · h_k )  ∈ R^{C+1}

  Eq. (15): pseudo-label set for class c at step n:
      P_c^n = { k : p_{n-1}(k)[c] is in top-θ% of { p_{n-1}(j)[c] }_j }
      N_c^n = { k : p_{n-1}(k)[c] is in bottom-θ% of { p_{n-1}(j)[c] }_j }

  Eq. (16): refinement cross-entropy loss:
      L_ref = (1/N) ∑_n ∑_c CE( p_n(P_c^n), c )
            + (1/N) ∑_n ∑_c CE( p_n(N_c^n), C )

References
----------
* Gao et al. (2025) — SMMILe, Nature Cancer (§3.3 Instance Refinement).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InstanceRefinement(nn.Module):
    """N independent linear refinement layers for per-instance prediction.

    Each layer v_n : R^{D_in} → R^{C+1} is an independent
    ``nn.Linear(in_dim, n_classes + 1)`` without shared parameters.
    This allows each layer to be trained with progressively refined
    pseudo-labels from the previous layer's output.

    Parameters
    ----------
    in_dim:
        Dimensionality of the input NIC instance features (= nic_out_channels
        in SMMILe, typically 256).
    n_classes:
        Number of cancer subtypes C.  Output dimension is C+1 (including
        the background / non-informative class at index C).
    n_layers:
        Number of refinement layers N.  The paper uses N = 3.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 5,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        out_dim = n_classes + 1   # C cancer subtypes + 1 background

        # v_1, v_2, ..., v_N  — each maps R^{D_in} → R^{C+1}
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=True) for _ in range(n_layers)]
        )

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Apply all N refinement layers to the instance feature matrix.

        All N layers receive the *same* input h (the NIC features); the
        progressive supervision is applied externally during training.

        Parameters
        ----------
        h : (K, in_dim)
            NIC instance features for all K patches in one WSI.

        Returns
        -------
        logits_list : list of N tensors, each (K, n_classes + 1)
            Raw (pre-softmax) logits from each refinement layer.
            ``logits_list[n]`` corresponds to v_{n+1}(h) in the paper.
            The Trainer uses ``logits_list[-1]`` (last layer) for inference
            and heatmap generation; earlier layers are used during the
            progressive pseudo-label training schedule.
        """
        return [layer(h) for layer in self.layers]

    # ------------------------------------------------------------------
    # Pseudo-label selection (Eq. 15 in SMMILe §3.3)
    # ------------------------------------------------------------------

    @staticmethod
    def select_pseudo_labels(
        scores: torch.Tensor,
        n_classes: int,
        theta: float = 0.10,
    ) -> torch.Tensor:
        """Select pseudo-labels from the previous layer's softmax predictions.

        Implements the top-θ% / bottom-θ% selection strategy (SMMILe §3.3,
        Eq. 15).

        Strategy
        --------
        For each class c independently:
          • Top-θ% instances (highest ``scores[:, c]``) → positive label c.
          • Conflict resolution: if an instance is in the top-θ% for two
            classes, it keeps the label of the class with the higher score.

        After positive labelling:
          • Of the *remaining unassigned* instances, the bottom-θ% by max-class
            probability → negative label = n_classes (background).
          • Everything else → label -1 (ignore in CE loss).

        Parameters
        ----------
        scores : (K, n_classes + 1)
            Softmax probabilities from one refinement layer (or from Stage 1
            attention for the first layer's pseudo-labels).
            Only the first ``n_classes`` columns are used for class selection;
            column ``n_classes`` (background) is ignored for label generation.
        n_classes:
            Number of foreground cancer subtypes C.
        theta:
            Fraction of instances to label per class (default 0.10 = 10%).

        Returns
        -------
        pseudo_labels : (K,)
            Long tensor with values in {0 .. n_classes, -1}.
              • 0 .. n_classes-1 : positive label for that cancer subtype
              • n_classes         : negative (background) label
              • -1                : ignore (not used in the loss)
        """
        K = scores.shape[0]
        n_sel = max(1, int(K * theta))
        pseudo = scores.new_full((K,), fill_value=-1, dtype=torch.long)
        # Confidence of the current positive assignment per instance
        # (used to resolve conflicts: highest-confidence class wins)
        conf = scores.new_zeros(K)

        # --- Positive labels: top-θ% per class
        for c in range(n_classes):
            probs_c = scores[:, c]                       # (K,)
            top_vals, top_idx = probs_c.topk(n_sel)     # both (n_sel,)

            # Assign label c only if no existing label has higher confidence
            beat_existing = top_vals > conf[top_idx]
            assign_idx = top_idx[beat_existing]
            pseudo[assign_idx] = c
            conf[assign_idx] = top_vals[beat_existing]

        # --- Negative labels: bottom-θ% of unassigned instances (by max score)
        unassigned = pseudo == -1
        if unassigned.any():
            max_prob, _ = scores[:, :n_classes].max(dim=-1)  # (K,)
            # Only consider unassigned instances for background selection
            max_prob_ua = max_prob.clone()
            max_prob_ua[~unassigned] = float("inf")           # exclude assigned
            n_neg = max(1, int(unassigned.sum().item() * theta))
            bot_idx = max_prob_ua.topk(n_neg, largest=False).indices
            # Assign background only to still-unassigned instances
            still_unset = pseudo[bot_idx] == -1
            pseudo[bot_idx[still_unset]] = n_classes          # background

        return pseudo
