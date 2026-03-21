"""Training logging utilities.

:class:`TrainingLogger` provides a unified interface for writing metrics to
both CSV files and TensorBoard event files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """CSV + TensorBoard logger.

    Parameters
    ----------
    log_dir : str | Path
        Directory for TensorBoard events and the CSV log file.
    """

    def __init__(self, log_dir: str | Path = "outputs/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "training_log.csv"
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        self._csv_initialised = False

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
    ) -> None:
        """Log metrics for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.
        train_loss : float
            Average training loss.
        val_metrics : Dict[str, float]
            Validation metrics dictionary.
        """
        # TensorBoard
        self.writer.add_scalar("loss/train", train_loss, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

        # CSV
        row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}"}
        row.update({k: f"{v:.6f}" for k, v in val_metrics.items()})

        if not self._csv_initialised:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialised = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar to TensorBoard.

        Parameters
        ----------
        tag : str
            Metric name.
        value : float
            Scalar value.
        step : int
            Global step.
        """
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
