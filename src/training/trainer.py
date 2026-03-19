"""
Training loop for SMMILe Stage 1 and Stage 2.

Handles:
  - Weighted sampling for class imbalance
  - Early stopping on validation loss
  - Checkpoint saving / loading
  - TensorBoard logging
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ..models.smmile import SMMILe
from .losses import ClsLoss, ConsLoss, MRFLoss, RefLoss
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Manages training for a single fold.

    Args:
        model:        SMMILe model instance.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        cfg:          Parsed YAML config dict (full config).
        output_dir:   Directory for checkpoints and TensorBoard logs.
        device:       torch.device.
    """

    def __init__(
        self,
        model: SMMILe,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        output_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        train_cfg = cfg["training"]
        loss_cfg = cfg["loss"]

        self.stage = train_cfg["stage"]
        self.epochs = train_cfg["epochs"]
        self.n_classes = cfg["model"]["n_classes"]

        # Optimizer
        opt_cfg = train_cfg["optimizer"]
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "none")
        if sched_name == "cosine":
            self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,
                    eta_min=sched_cfg.get("min_lr", 1e-7),
                )
            )
        else:
            self.scheduler = None

        # Loss functions
        self.cls_loss_fn = ClsLoss(n_classes=self.n_classes)
        self.ref_loss_fn = RefLoss(
            theta=loss_cfg.get("pseudo_label", {}).get("theta", 0.10),
            n_classes=self.n_classes,
        )
        mrf_cfg = loss_cfg.get("mrf", {})
        self.mrf_loss_fn = MRFLoss(
            lambda1=mrf_cfg.get("lambda1", 0.8),
            lambda2=mrf_cfg.get("lambda2", 0.2),
            n_classes=self.n_classes,
        )
        self.cons_loss_fn = ConsLoss()
        self.use_consistency = loss_cfg.get("consistency", False)

        self.loss_weights = {
            "cls": loss_cfg.get("cls_weight", 1.0),
            "ref": loss_cfg.get("ref_weight", 1.0),
            "mrf": loss_cfg.get("mrf_weight", 1.0),
            "cons": loss_cfg.get("cons_weight", 0.0),
        }

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 10)
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        # Logging
        log_cfg = cfg.get("logging", {})
        self.log_every_n = log_cfg.get("log_every_n_steps", 10)
        tb_dir = self.output_dir / "tensorboard"
        self.writer: SummaryWriter | None = (
            SummaryWriter(str(tb_dir)) if log_cfg.get("tensorboard", True) else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop for all epochs."""
        logger.info("Starting Stage %d training for %d epochs.", self.stage, self.epochs)
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                epoch, self.epochs, train_loss, val_loss,
            )

            improved = self._checkpoint(val_loss, epoch)
            if not improved:
                self._patience_counter += 1
                if self.early_stopping_enabled and self._patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break
            else:
                self._patience_counter = 0

        if self.writer:
            self.writer.close()
        logger.info("Training complete. Best val loss: %.4f", self._best_val_loss)

    def load_stage1_checkpoint(self, ckpt_path: str | Path) -> None:
        """Load Stage 1 weights before Stage 2 training."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded Stage 1 checkpoint from %s "
            "(missing=%d, unexpected=%d keys).",
            ckpt_path, len(missing), len(unexpected),
        )

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            embeddings = batch["embeddings"].squeeze(0).to(self.device)  # (N, D)
            superpixels = batch["superpixels"].squeeze(0).to(self.device)  # (N,)
            label = batch["label"].item()

            self.optimizer.zero_grad()
            out = self.model(embeddings, superpixels)
            loss = self._compute_loss(out, label, superpixels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.writer and step % self.log_every_n == 0:
                global_step = (epoch - 1) * len(self.train_loader) + step
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            embeddings = batch["embeddings"].squeeze(0).to(self.device)
            superpixels = batch["superpixels"].squeeze(0).to(self.device)
            label = batch["label"].item()
            out = self.model(embeddings, superpixels)
            loss = self._compute_loss(out, label, superpixels)
            total_loss += loss.item()
        return total_loss / max(1, len(self.val_loader))

    def _compute_loss(
        self,
        out: dict,
        label: int,
        superpixels: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.loss_weights["cls"] * self.cls_loss_fn(out["bag_logits"], label)

        if self.stage >= 2 and "ref_logits" in out:
            loss = loss + self.loss_weights["ref"] * self.ref_loss_fn(
                out["ref_logits"], out["attn"], label
            )
            loss = loss + self.loss_weights["mrf"] * self.mrf_loss_fn(
                out["ref_logits"], superpixels
            )

        if self.use_consistency and "ref_logits" in out:
            loss = loss + self.loss_weights["cons"] * self.cons_loss_fn(
                out["ref_logits"], label
            )

        return loss

    def _checkpoint(self, val_loss: float, epoch: int) -> bool:
        """Save checkpoint if val_loss improved. Returns True if improved."""
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.cfg,
            }
            best_path = self.output_dir / "best_model.pth"
            torch.save(ckpt, best_path)
            logger.debug("Saved best checkpoint → %s (val_loss=%.4f)", best_path, val_loss)
            return True
        # Always save last
        last_path = self.output_dir / "last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss,
                "config": self.cfg,
            },
            last_path,
        )
        return False
