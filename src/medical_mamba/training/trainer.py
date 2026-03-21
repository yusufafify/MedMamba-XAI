"""Trainer — encapsulates train / val / test loops for MedicalVMamba.

Merge decisions
---------------
Config         : ``TrainConfig`` dataclass (website) over raw YAML dict
                 (agent).  Type-safe, IDE-autocomplete, no silent KeyErrors.
                 The YAML loading lives in ``scripts/train.py``; ``Trainer``
                 receives a fully-resolved ``TrainConfig``.

Optimizer      : Three param groups (website): backbone at ``lr``, heads at
                 ``5×lr``, Kendall params at ``0.1×lr``.  Single group
                 (agent) throws away the most valuable training trick for
                 fine-tuning-style architectures.

Batch format   : Dict ``{image, label, task_id}`` (website / merged dataset).
                 Agent's ``len(batch) == 2 or 3`` check is fragile and breaks
                 silently if the dataset changes return shape.

Kendall logging: Σ values logged to TensorBoard each epoch (website).
                 Essential diagnostic — a σ that stays near 1.0 means the
                 model treats that task as equally hard throughout; a growing
                 σ means it's deprioritising it.

Checkpoint     : ``latest + best`` with ``criterion`` state included (website).
                 Agent saved best-only and dropped Kendall params — resuming
                 would reinitialise σ values to 0, breaking the weighting.

Early stopping : ``patience`` counter on ``avg_f1_macro`` (agent logic, kept).
                 Website stopped on ``val_loss`` which is less interpretable
                 for a multi-task model.

autocast       : Uses ``torch.amp.autocast(device_type=...)`` (new API) over
                 the deprecated ``torch.cuda.amp.autocast``.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from medical_mamba.data.constants import DATASET_META
from medical_mamba.models.medical_vmamba import MedicalVMamba, build_model
from medical_mamba.training.losses import KendallMultiTaskLoss
from medical_mamba.training.metrics import TaskMetricTracker
from medical_mamba.training.schedulers import CosineWarmupScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Fully typed training configuration.

    All fields have sensible defaults so a minimal script can do::

        cfg = TrainConfig(single_dataset="pathmnist")
        trainer = Trainer(cfg, train_loader, val_loader, test_loader)
        trainer.fit()
    """

    # ── Data ──────────────────────────────────────────────────────────────
    dataset_roots: Dict[str, str] = field(default_factory=lambda: {
        "pathmnist":  "D:/medmnist/pathmnist",
        "bloodmnist": "D:/medmnist/bloodmnist",
        "dermamnist": "D:/medmnist/dermamnist",
        "octmnist":   "D:/medmnist/octmnist",
    })
    single_dataset: Optional[str] = None     # set to "pathmnist" etc. for safety-net mode

    # ── Model ─────────────────────────────────────────────────────────────
    model_size:   str   = "tiny"             # "tiny" | "small" | "base"
    patch_size:   int   = 4                  # 4 = high-res, 8 = fast/low-VRAM
    head_dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────
    epochs:          int   = 100
    lr:              float = 1e-4
    weight_decay:    float = 0.05
    warmup_epochs:   int   = 10
    grad_clip:       float = 1.0
    label_smoothing: float = 0.1
    use_amp:         bool  = True
    patience:        int   = 15             # early-stopping epochs without improvement

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str          = "./runs/medical_mamba"
    resume:     Optional[str] = None        # path to checkpoint to resume from

    # ── Hardware ──────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml_dict(cls, d: Dict) -> "TrainConfig":
        """Construct from a merged YAML config dict (``scripts/train.py``)."""
        flat: Dict = {}
        # Flatten nested YAML sections into the dataclass fields
        for section in d.values():
            if isinstance(section, dict):
                flat.update(section)
            # top-level scalar keys
        flat.update({k: v for k, v in d.items() if not isinstance(v, dict)})
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in flat.items() if k in valid})


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Unified trainer for single-task and multi-task VMamba training.

    Parameters
    ----------
    cfg : TrainConfig
        Resolved training configuration.
    train_loader : DataLoader
        Training data loader (built by ``dataset.build_dataloaders``).
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        train_loader,
        val_loader,
        test_loader,
    ) -> None:
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.amp_device = "cuda" if "cuda" in cfg.device else "cpu"

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # ── Active tasks ──────────────────────────────────────────────────
        if cfg.single_dataset is not None:
            active = {cfg.single_dataset: cfg.dataset_roots[cfg.single_dataset]}
        else:
            active = cfg.dataset_roots

        self.task_names: List[str] = list(active.keys())
        self.task_configs: List[Tuple[str, int]] = [
            (name, DATASET_META[name]["num_classes"])
            for name in self.task_names
        ]

        # ── DataLoaders ───────────────────────────────────────────────────
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # ── Model ─────────────────────────────────────────────────────────
        self.log(f"Building VMamba-{cfg.model_size} | patch={cfg.patch_size}")
        self.model = build_model(
            task_configs=self.task_configs,
            model_size=cfg.model_size,
            patch_size=cfg.patch_size,
            head_dropout=cfg.head_dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.log(f"Parameters: {n_params:.1f}M | Tasks: {self.task_names}")

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = KendallMultiTaskLoss(
            task_names=self.task_names,
            label_smoothing=cfg.label_smoothing,
        ).to(self.device)

        # ── Optimizer: 3 param groups ─────────────────────────────────────
        # Backbone: standard lr + weight decay
        # Heads:    5× lr  (need to adapt faster to task-specific features)
        # Kendall:  0.1× lr (σ params should move slowly)
        backbone_params = list(self.model.backbone.parameters())
        head_params     = [
            p for n, p in self.model.named_parameters()
            if "backbone" not in n and "criterion" not in n
        ]
        kendall_params  = list(self.criterion.parameters())

        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr,         "weight_decay": cfg.weight_decay},
            {"params": head_params,     "lr": cfg.lr * 5.0,   "weight_decay": 0.0},
            {"params": kendall_params,  "lr": cfg.lr * 0.1,   "weight_decay": 0.0},
        ])

        # ── Scheduler ─────────────────────────────────────────────────────
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.warmup_epochs,
            max_epochs=cfg.epochs,
            min_lr=1e-6,
        )

        # ── AMP ───────────────────────────────────────────────────────────
        self.scaler = GradScaler(
            device=self.amp_device,
            enabled=(cfg.use_amp and "cuda" in cfg.device),
        )

        # ── Logging ───────────────────────────────────────────────────────
        self.writer       = SummaryWriter(self.out_dir / "tensorboard")
        self.csv_path     = self.out_dir / "metrics.csv"
        self.metric_tracker = TaskMetricTracker(task_names=self.task_names)

        # ── State ─────────────────────────────────────────────────────────
        self.start_epoch              = 0
        self.best_avg_f1: float       = 0.0
        self.epochs_no_improve: int   = 0

        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    # ------------------------------------------------------------------ #
    #  Logging setup                                                       #
    # ------------------------------------------------------------------ #

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.out_dir / "train.log"),
                logging.StreamHandler(),
            ],
        )
        self._logger = logging.getLogger("MedMamba")

    def log(self, msg: str) -> None:
        self._logger.info(msg)

    # ------------------------------------------------------------------ #
    #  Core epoch                                                          #
    # ------------------------------------------------------------------ #

    def _run_epoch(self, split: str) -> Dict:
        """Run one full pass over ``split`` ∈ {train, val, test}.

        Returns
        -------
        Dict
            ``{total_loss, task_losses, metrics}``
        """
        loader   = {"train": self.train_loader,
                    "val":   self.val_loader,
                    "test":  self.test_loader}[split]
        is_train = split == "train"

        self.model.train(is_train)
        self.metric_tracker.reset()

        total_loss   = 0.0
        all_task_losses: Dict[str, List[float]] = {t: [] for t in self.task_names}
        n_batches    = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc=f"[{split}]", leave=False)
            for batch in pbar:
                # Batch dict guaranteed by merged dataset.py
                images   = batch["image"].to(self.device, non_blocking=True)
                labels   = batch["label"].to(self.device, non_blocking=True)
                task_ids = batch["task_id"].to(self.device, non_blocking=True)

                with autocast(device_type=self.amp_device,
                              enabled=(self.cfg.use_amp and "cuda" in self.cfg.device)):
                    if len(self.task_names) == 1:
                        # Single-task: cleaner forward path
                        logits, _ = self.model.forward_single(images, self.task_names[0])
                        task_logits = {self.task_names[0]: logits}
                    else:
                        task_logits, _ = self.model.forward_multi(images, task_ids)

                    loss, task_loss_dict = self.criterion(task_logits, labels, task_ids)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        self.cfg.grad_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Accumulate per-task losses for logging
                for name, v in task_loss_dict.items():
                    all_task_losses[name].append(v)

                # Metrics
                self.metric_tracker.update_multitask(
                    task_logits, labels, task_ids,
                    task_losses=task_loss_dict,
                )

                total_loss += loss.item()
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        metrics = self.metric_tracker.compute()
        avg_task_losses = {
            name: float(sum(vs) / max(len(vs), 1))
            for name, vs in all_task_losses.items()
        }
        return {
            "total_loss": total_loss / max(n_batches, 1),
            "task_losses": avg_task_losses,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------ #
    #  Main fit loop                                                       #
    # ------------------------------------------------------------------ #

    def fit(self) -> Dict:
        """Run full training + return best checkpoint info."""
        self.log("=" * 64)
        self.log(f"MedicalVMamba-{self.cfg.model_size} | "
                 f"tasks={self.task_names} | device={self.device} | "
                 f"AMP={self.cfg.use_amp}")
        self.log("=" * 64)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()

            train_out = self._run_epoch("train")
            self.scheduler.step()
            val_out   = self._run_epoch("val")

            elapsed   = time.time() - t0
            val_m     = val_out["metrics"]
            avg_f1    = val_m.get("avg_f1_macro", val_m.get("f1_macro", 0.0))

            # ── Console ───────────────────────────────────────────────────
            self.log(
                f"Ep {epoch+1:03d}/{self.cfg.epochs} | "
                f"train_loss={train_out['total_loss']:.4f} | "
                f"val_loss={val_out['total_loss']:.4f} | "
                f"avg_f1={avg_f1:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.0f}s"
            )
            for name in self.task_names:
                acc = val_m.get(f"{name}_accuracy", val_m.get("accuracy", 0))
                f1  = val_m.get(f"{name}_f1_macro", val_m.get("f1_macro", 0))
                self.log(f"  [{name}] acc={acc:.4f} f1={f1:.4f}")

            # ── Kendall σ diagnostic ──────────────────────────────────────
            for name, sigma in self.criterion.sigma_values().items():
                self.log(f"  [sigma] {name}: {sigma:.4f}")
                self.writer.add_scalar(f"kendall_sigma/{name}", sigma, epoch)

            # ── TensorBoard ───────────────────────────────────────────────
            self.writer.add_scalars(
                "loss/total",
                {"train": train_out["total_loss"], "val": val_out["total_loss"]},
                epoch,
            )
            for name in self.task_names:
                self.writer.add_scalar(
                    f"val/{name}_f1",
                    val_m.get(f"{name}_f1_macro", val_m.get("f1_macro", 0)),
                    epoch,
                )
            self.writer.add_scalar("val/avg_f1_macro", avg_f1, epoch)

            # ── CSV ───────────────────────────────────────────────────────
            self._log_csv(epoch, "train", train_out)
            self._log_csv(epoch, "val",   val_out)

            # ── Checkpoint + early stopping ───────────────────────────────
            is_best = avg_f1 > self.best_avg_f1
            if is_best:
                self.best_avg_f1 = avg_f1
                self.epochs_no_improve = 0
                self.log(f"  >> New best avg_f1={avg_f1:.4f}")
            else:
                self.epochs_no_improve += 1

            self._save_checkpoint(epoch, is_best)

            if self.epochs_no_improve >= self.cfg.patience:
                self.log(f"Early stopping at epoch {epoch+1} "
                         f"(no improvement for {self.cfg.patience} epochs)")
                break

        self.log("Training complete.")
        self.writer.close()
        return {
            "best_avg_f1": self.best_avg_f1,
            "checkpoint":  str(self.out_dir / "checkpoint_best.pt"),
        }

    # ------------------------------------------------------------------ #
    #  Test evaluation                                                     #
    # ------------------------------------------------------------------ #

    def evaluate_test(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Load best checkpoint and run test-set evaluation."""
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        elif (self.out_dir / "checkpoint_best.pt").exists():
            self._load_checkpoint(str(self.out_dir / "checkpoint_best.pt"))

        test_out = self._run_epoch("test")
        self.log("\n=== TEST RESULTS ===")
        for name in self.task_names:
            acc = test_out["metrics"].get(f"{name}_accuracy", test_out["metrics"].get("accuracy", 0))
            f1  = test_out["metrics"].get(f"{name}_f1_macro", test_out["metrics"].get("f1_macro", 0))
            self.log(f"  [{name}] acc={acc:.4f} f1={f1:.4f}")
        return test_out

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "scheduler_state_dict":self.scheduler.state_dict(),
            "criterion_state_dict":self.criterion.state_dict(),   # saves σ params
            "best_avg_f1":         self.best_avg_f1,
            "task_names":          self.task_names,
            "config":              asdict(self.cfg),
        }
        torch.save(state, self.out_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(state, self.out_dir / "checkpoint_best.pt")

    def _load_checkpoint(self, path: str) -> None:
        self.log(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.criterion.load_state_dict(state["criterion_state_dict"])
        self.best_avg_f1  = state.get("best_avg_f1", 0.0)
        self.start_epoch  = state.get("epoch", 0) + 1
        self.log(f"Resumed from epoch {self.start_epoch}")

    # ------------------------------------------------------------------ #
    #  CSV logging                                                         #
    # ------------------------------------------------------------------ #

    def _log_csv(self, epoch: int, split: str, epoch_out: Dict) -> None:
        row: Dict = {
            "epoch": epoch,
            "split": split,
            "total_loss": epoch_out["total_loss"],
        }
        row.update({f"task_loss_{k}": v for k, v in epoch_out["task_losses"].items()})
        row.update(epoch_out["metrics"])

        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
