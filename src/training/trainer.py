"""
SMMILeTrainer — end-to-end training and evaluation for the SMMILe framework.

Handles two-stage MIL training for ovarian cancer WSI subtype classification:

  Stage 1: train primary network (NIC + GatedAttention + InD/InS) with L_cls.
  Stage 2: load Stage 1 weights, add InstanceRefinement, train with
           L_cls + L_ref + L_mrf.
  Evaluate: compute WSI-level macro AUC and patch-level spatial metrics;
            save per-slide instance prediction CSV for heatmap generation.

Paper: Gao et al. (2025) — SMMILe, Nature Cancer.
       DOI: 10.1038/s43018-025-01060-8
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ..models.smmile import SMMILe
from ..datasets.mil_dataset import MILDataset
from .losses import SMMILeTotalLoss
from .evaluator import Evaluator, EvalResult

logger = logging.getLogger(__name__)

# Human-readable class names (index → abbreviation)
CLASS_NAMES = ["CC", "EC", "HGSC", "LGSC", "MC"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SMMILeTrainer:
    """Two-stage MIL trainer for SMMILe ovarian cancer subtype classification.

    Parameters
    ----------
    model:
        Pre-built :class:`SMMILe` instance, or ``None`` to build from *config*.
        If provided and the architecture does not match the stage being run,
        the trainer will rebuild internally (e.g. adding refinement layers for
        Stage 2).
    config:
        Full YAML config dict (e.g. loaded from ``ovarian_conch_s1.yaml`` or
        ``ovarian_conch_s2.yaml``).  Keys used:
        ``model``, ``training``, ``loss``, ``paths``, ``hardware``,
        ``logging``.
    fold_idx:
        Current cross-validation fold index (0-based).  Used to name output
        directories and checkpoint files.
    """

    def __init__(
        self,
        model: SMMILe | None,
        config: dict,
        fold_idx: int,
    ) -> None:
        self.config    = config
        self.fold_idx  = fold_idx
        self.n_classes = config["model"]["n_classes"]

        # ---- Device -------------------------------------------------------
        hw      = config.get("hardware", {})
        gpu_id  = hw.get("gpu_id", 0)
        if torch.cuda.is_available() and gpu_id >= 0:
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

        # ---- Output directory (fold-specific) ----------------------------
        base_out       = Path(config["paths"]["output_dir"])
        self.output_dir = base_out / f"fold{fold_idx}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Model (may be None; built lazily by train_stage*) -----------
        self.model: SMMILe | None = model
        if self.model is not None:
            self.model = self.model.to(self.device)

        # Best checkpoint path — set after each train_stage* call
        self._best_ckpt: Path | None = None

        # Logging
        log_cfg = config.get("logging", {})
        self._log_every_n = log_cfg.get("log_every_n_steps", 10)
        self._use_tb      = log_cfg.get("tensorboard", True)

    # ======================================================================
    # Public API
    # ======================================================================

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Path:
        """Train Stage 1 (primary network, L_cls only).

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders yielding :class:`MILDataset` batches with keys
            ``embeddings``, ``superpixels``, ``coords``, ``label``,
            ``slide_id``.

        Returns
        -------
        Path to the best validation-loss checkpoint (``best_model.pth``).
        """
        logger.info("=== Stage 1 training — fold %d ===", self.fold_idx)
        self.model = self._build_model(stage=1).to(self.device)
        return self._run_stage(1, train_loader, val_loader)

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stage1_model_path: str | Path,
    ) -> Path:
        """Train Stage 2 (refinement + MRF, joint fine-tuning).

        Loads Stage 1 weights as initialisation, then jointly trains all
        components (NIC, attention, refinement layers) with L_cls + L_ref +
        L_mrf.

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders (same format as Stage 1).
        stage1_model_path:
            Path to Stage 1 ``best_model.pth`` checkpoint.

        Returns
        -------
        Path to the best validation-loss checkpoint.
        """
        logger.info("=== Stage 2 training — fold %d ===", self.fold_idx)
        self.model = self._build_model(stage=2).to(self.device)
        self._load_weights(stage1_model_path, strict=False)
        return self._run_stage(2, train_loader, val_loader)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate the current model on *test_loader*.

        Computes:
          - WSI-level macro AUC (bag logits → softmax)
          - Patch-level macro AUC / F1 / accuracy / precision / recall
            (Stage 2: ``ref_logits``; Stage 1 fallback: ``inst_scores``)

        Saves a per-slide instance prediction CSV to
        ``{output_dir}/inst_predictions_fold{fold_idx}.csv`` with columns:
        ``slide_id, x, y, predicted_class, prob_CC, prob_EC, prob_HGSC,
        prob_LGSC, prob_MC``

        Also writes ``{output_dir}/eval_metrics.json``.

        Parameters
        ----------
        test_loader:
            DataLoader over the test split.  Expects batch_size=1.

        Returns
        -------
        dict with scalar metric values (floats) and ``inst_csv_path`` (str).
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded.  Call train_stage1/2 first, or use "
                "load_checkpoint() before evaluate()."
            )

        self.model.eval()
        use_refinement = self.model.refinement is not None
        evaluator = Evaluator(n_classes=self.n_classes)
        inst_rows: list[dict] = []

        n_classes     = self.n_classes
        class_cols    = [f"prob_{cls}" for cls in CLASS_NAMES[:n_classes]]

        for batch in test_loader:
            slide_id   = batch["slide_id"][0]
            embeddings = batch["embeddings"].squeeze(0).to(self.device)   # (K, D)
            superpixels= batch["superpixels"].squeeze(0).to(self.device)  # (K,)
            coords     = batch["coords"].squeeze(0).cpu().numpy()          # (K, 2)
            label      = int(batch["label"].item())

            out = self.model(embeddings, superpixels)

            # ---- WSI-level bag prediction --------------------------------
            bag_probs = F.softmax(out["bag_logits"], dim=-1).cpu().numpy()
            evaluator.update_wsi(slide_id, bag_probs, label)

            # ---- Patch-level predictions ---------------------------------
            if use_refinement and "ref_logits" in out:
                # Stage 2: (K, C+1) → softmax over C cancer classes
                ref_logits = out["ref_logits"]               # (K, C+1)
                patch_probs_t = F.softmax(ref_logits[:, :n_classes], dim=-1)
            else:
                # Stage 1 fallback: inst_scores (K, C)
                patch_probs_t = F.softmax(out["inst_scores"], dim=-1)

            patch_probs  = patch_probs_t.cpu().numpy()       # (K, C)
            patch_preds  = patch_probs.argmax(axis=1)        # (K,)
            patch_labels = np.full(len(patch_preds), label)  # all patches = WSI label
            evaluator.update_patch(patch_probs, patch_labels)

            # ---- Instance CSV rows ---------------------------------------
            for k in range(len(patch_preds)):
                row: dict = {
                    "slide_id":       slide_id,
                    "x":              int(coords[k, 0]),
                    "y":              int(coords[k, 1]),
                    "predicted_class": CLASS_NAMES[patch_preds[k]],
                }
                for ci, col in enumerate(class_cols):
                    row[col] = float(patch_probs[k, ci])
                inst_rows.append(row)

        # ---- Compute metrics --------------------------------------------
        result: EvalResult = evaluator.compute()

        # ---- Save inst CSV ----------------------------------------------
        csv_path = self.output_dir / f"inst_predictions_fold{self.fold_idx}.csv"
        if inst_rows:
            pd.DataFrame(inst_rows).to_csv(csv_path, index=False)
            logger.info("Saved instance predictions → %s (%d rows)", csv_path, len(inst_rows))

        # ---- Save metrics JSON ------------------------------------------
        metrics: dict = {
            "fold":            self.fold_idx,
            "wsi_auc":         result.wsi_auc,
            "patch_auc":       result.patch_auc,
            "patch_f1":        result.patch_f1,
            "patch_acc":       result.patch_acc,
            "patch_precision": result.patch_precision,
            "patch_recall":    result.patch_recall,
            "inst_csv_path":   str(csv_path),
        }
        json_path = self.output_dir / "eval_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "Evaluation fold %d: WSI AUC=%.4f | Patch AUC=%.4f F1=%.4f Acc=%.4f",
            self.fold_idx,
            result.wsi_auc, result.patch_auc, result.patch_f1, result.patch_acc,
        )
        return metrics

    def load_checkpoint(self, ckpt_path: str | Path, stage: int = 2) -> None:
        """Build model for *stage* and load weights from *ckpt_path*.

        Useful for standalone evaluation (``06_evaluate.sh``) where training
        was run in a previous session.
        """
        self.model = self._build_model(stage).to(self.device)
        self._load_weights(ckpt_path, strict=True)
        self._best_ckpt = Path(ckpt_path)
        logger.info("Loaded checkpoint from %s", ckpt_path)

    # ======================================================================
    # Weighted sampling helper (called from train.py / test code)
    # ======================================================================

    @staticmethod
    def build_weighted_sampler(dataset: MILDataset) -> WeightedRandomSampler:
        """Return a :class:`WeightedRandomSampler` that up-samples rare classes.

        Weights are inversely proportional to class frequency so each epoch
        sees approximately equal representation across subtypes.
        """
        labels = [dataset.labels[sid] for sid in dataset.slide_ids]
        counts  = Counter(labels)
        weights = [1.0 / counts[lab] for lab in labels]
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    # ======================================================================
    # Private: model construction
    # ======================================================================

    def _build_model(self, stage: int) -> SMMILe:
        """Instantiate SMMILe from config for the given training stage."""
        mcfg = self.config["model"]
        nic  = mcfg.get("nic",      {})
        attn = mcfg.get("attention", {})
        ind  = mcfg.get("instance_dropout", {})
        ins  = mcfg.get("instance_sampling", {})
        ref  = mcfg.get("refinement", {})

        n_ref_layers = ref.get("n_layers", 3) if stage >= 2 else 0

        model = SMMILe(
            embedding_dim      = mcfg.get("embedding_dim", 512),
            n_classes          = mcfg.get("n_classes", 5),
            nic_out_channels   = nic.get("out_channels", 256),
            nic_kernel_size    = nic.get("kernel_size", 3),
            attn_hidden_dim    = attn.get("hidden_dim", 256),
            attn_dropout       = attn.get("dropout", True),
            attn_dropout_rate  = attn.get("dropout_rate", 0.25),
            ind_drop_rate      = ind.get("drop_rate", 0.5) if ind.get("enabled", True) else 0.0,
            n_refinement_layers= n_ref_layers,
            ins_enabled        = ins.get("enabled", True),
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Built SMMILe Stage %d — %d trainable parameters "
            "(refinement layers: %d)",
            stage, n_params, n_ref_layers,
        )
        return model

    # ======================================================================
    # Private: optimiser / scheduler / loss
    # ======================================================================

    def _build_optimizer(self, model: SMMILe) -> torch.optim.Adam:
        opt_cfg = self.config["training"]["optimizer"]
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = opt_cfg.get("lr", 2e-5),
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
            betas        = tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

    def _build_scheduler(self, optimizer, n_epochs: int):
        sched_cfg  = self.config["training"].get("scheduler", {})
        sched_name = sched_cfg.get("name", "none")
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = n_epochs,
                eta_min = sched_cfg.get("min_lr", 1e-7),
            )
        return None

    def _build_loss(self, stage: int) -> SMMILeTotalLoss:
        lcfg = self.config.get("loss", {})
        mrf  = lcfg.get("mrf", {})
        pl   = lcfg.get("pseudo_label", {})
        return SMMILeTotalLoss(
            n_classes        = self.n_classes,
            theta            = pl.get("theta", 0.10),
            lambda1          = mrf.get("lambda1", 0.8),
            lambda2          = mrf.get("lambda2", 0.2),
            cls_weight       = lcfg.get("cls_weight", 1.0),
            ref_weight       = lcfg.get("ref_weight", 1.0),
            mrf_weight       = lcfg.get("mrf_weight", 1.0),
            cons_weight      = lcfg.get("cons_weight", 0.0),
            stage            = stage,
        )

    # ======================================================================
    # Private: core training loop
    # ======================================================================

    def _run_stage(
        self,
        stage: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Path:
        """Inner loop shared by train_stage1 and train_stage2."""
        assert self.model is not None

        train_cfg  = self.config["training"]
        es_cfg     = train_cfg.get("early_stopping", {})
        n_epochs   = train_cfg.get("epochs", 40)
        patience   = es_cfg.get("patience", 10)
        es_enabled = es_cfg.get("enabled", True)

        optimizer  = self._build_optimizer(self.model)
        scheduler  = self._build_scheduler(optimizer, n_epochs)
        loss_fn    = self._build_loss(stage)

        best_val  = float("inf")
        no_improv = 0
        best_path = self.output_dir / "best_model.pth"
        last_path = self.output_dir / "last_model.pth"

        writer = None
        if self._use_tb:
            tb_dir = self.output_dir / "tensorboard"
            writer = SummaryWriter(str(tb_dir))

        for epoch in range(1, n_epochs + 1):
            train_loss, train_ld = self._train_epoch(
                self.model, train_loader, optimizer, loss_fn, stage
            )
            val_loss = self._val_epoch(
                self.model, val_loader, loss_fn, stage
            )

            if scheduler is not None:
                scheduler.step()

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val",   val_loss,   epoch)
                for k, v in train_ld.items():
                    if k != "total":
                        writer.add_scalar(f"Loss/train_{k}", v.item(), epoch)

            logger.info(
                "Stage %d | Epoch %3d/%d | train=%.4f  val=%.4f",
                stage, epoch, n_epochs, train_loss, val_loss,
            )

            if val_loss < best_val:
                best_val  = val_loss
                no_improv = 0
                torch.save(
                    {
                        "epoch":              epoch,
                        "stage":              stage,
                        "model_state_dict":   self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss":           val_loss,
                        "config":             self.config,
                    },
                    best_path,
                )
                logger.debug(
                    "  ✓ Best checkpoint saved (val_loss=%.4f)", val_loss
                )
            else:
                no_improv += 1
                torch.save(
                    {
                        "epoch":            epoch,
                        "stage":            stage,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss":         val_loss,
                        "config":           self.config,
                    },
                    last_path,
                )

            if es_enabled and no_improv >= patience:
                logger.info(
                    "  Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, patience,
                )
                break

        if writer:
            writer.close()

        logger.info(
            "Stage %d training complete — best val loss: %.4f → %s",
            stage, best_val, best_path,
        )
        self._best_ckpt = best_path
        return best_path

    def _train_epoch(
        self,
        model: SMMILe,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: SMMILeTotalLoss,
        stage: int,
    ) -> tuple[float, dict]:
        model.train()
        total_loss = 0.0
        last_ld: dict = {}

        for step, batch in enumerate(loader):
            embeddings  = batch["embeddings"].squeeze(0).to(self.device)   # (K, D)
            superpixels = batch["superpixels"].squeeze(0).to(self.device)  # (K,)
            coords      = batch["coords"].squeeze(0).to(self.device)       # (K, 2)
            label       = int(batch["label"].item())

            optimizer.zero_grad()
            out         = model(embeddings, superpixels)
            loss, ld    = loss_fn(out, label, superpixels, coords)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_ld     = ld

        n = max(1, len(loader))
        return total_loss / n, {k: v.detach() for k, v in last_ld.items()}

    @torch.no_grad()
    def _val_epoch(
        self,
        model: SMMILe,
        loader: DataLoader,
        loss_fn: SMMILeTotalLoss,
        stage: int,
    ) -> float:
        model.eval()
        total_loss = 0.0
        for batch in loader:
            embeddings  = batch["embeddings"].squeeze(0).to(self.device)
            superpixels = batch["superpixels"].squeeze(0).to(self.device)
            coords      = batch["coords"].squeeze(0).to(self.device)
            label       = int(batch["label"].item())
            out         = model(embeddings, superpixels)
            loss, _     = loss_fn(out, label, superpixels, coords)
            total_loss += loss.item()
        return total_loss / max(1, len(loader))

    # ======================================================================
    # Private: checkpoint loading
    # ======================================================================

    def _load_weights(self, ckpt_path: str | Path, strict: bool = False) -> None:
        """Load model weights from *ckpt_path* into ``self.model``."""
        ckpt  = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        logger.info(
            "Loaded weights from %s "
            "(strict=%s, missing=%d, unexpected=%d keys)",
            ckpt_path, strict, len(missing), len(unexpected),
        )
        if missing:
            logger.debug("  Missing keys: %s", missing[:10])
