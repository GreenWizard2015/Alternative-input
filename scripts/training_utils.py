#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training utilities for model trainers (gaze prediction).

This module provides common functionality for training loops including:
- Progress display with ETA calculation
- Model checkpoint tracking and saving
- Early stopping with best model management
- Evaluation wrapper pattern
"""

import os
import time
import sys
from collections import defaultdict
from typing import Any, Callable, List, Dict, Tuple, Set

import numpy as np
from Core.logging_config import get_logger

logger = get_logger(__name__)

# Constants for progress display
DISPLAY_FREQUENCY = 10  # Display summary every N iterations
CLEAR_SCREEN_UNIX = "\033[2J\033[H"  # ANSI escape codes: clear screen and move to home
CLEAR_SCREEN_WIN = "\033c"  # Windows clear screen escape sequence


def format_time(total_sec: float) -> str:
    """Format seconds as d:h:m:s, h:m:s, or m:s depending on magnitude.

    Args:
        total_sec: Total seconds to format.

    Returns:
        Formatted time string (e.g., "1d:02h:30m:45s" or "45m:30s").

    Example:
        >>> format_time(3661)  # 1 hour, 1 minute, 1 second
        '1h:01m:01s'
        >>> format_time(90061)  # 1 day, 1 hour, 1 minute, 1 second
        '1d:01h:01m:01s'
    """
    days, remainder = divmod(total_sec, 86400)
    hours, remainder = divmod(remainder, 3600)
    mins, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d:{hours:02d}h:{mins:02d}m:{secs:02d}s"
    if hours > 0:
        return f"{hours}h:{mins:02d}m:{secs:02d}s"

    return f"{mins}m:{secs:02d}s"


def _mean_metrics(gen: Any) -> Dict[str, float]:
    """Compute mean of metrics from generator.

    Args:
        gen: Callable that yields dictionaries of metric values.

    Returns:
        Dictionary with mean values for each metric key.

    Example:
        >>> def metric_gen(): yield {"loss": 0.5}; yield {"loss": 0.3}
        >>> means = _mean_metrics(metric_gen)
        >>> assert means["loss"] == 0.4
    """
    metrics: Dict[str, List[float]] = defaultdict(list)
    for data in gen():
        for k, v in data.items():
            v = [v] if isinstance(v, np.float32) else v
            metrics[k].extend(v)
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def create_progress_display(desc: str, total_samples: int) -> Callable:
    """Create a progress display generator for training loops.

    Args:
        desc: Description string for progress display (e.g., "Epoch 1 / 10").
        total_samples: Total samples in epoch.

    Returns:
        Callable that formats progress display information (clears screen and prints).

    Raises:
        ValueError: If total_samples <= 0.

    Example:
        >>> formatter = create_progress_display("Epoch 1 / 10", 100)
        >>> formatter(50, ["loss=0.5"])  # Display at step 50
    """
    epoch_start_time = time.time()

    def format_progress_display(
        step: int,
        loss_items: List[str],
    ) -> None:
        """Format progress display with header, timing, and loss information.

        Clears terminal and prints progress bar with ETA calculation.

        Args:
            step: Current step number (0-indexed).
            loss_items: List of formatted loss strings (e.g., ["loss=0.5", "acc=0.9"]).
        """
        steps_completed = step + 1
        elapsed = time.time() - epoch_start_time
        remaining_steps = total_samples - step
        iter_per_sec = steps_completed / elapsed if elapsed > 0 else 0
        eta_sec = remaining_steps / iter_per_sec if iter_per_sec > 0 else 0

        # Clear terminal in a cross-platform way
        clear_code = CLEAR_SCREEN_UNIX if os.name == "posix" else CLEAR_SCREEN_WIN
        sys.stdout.write(clear_code)
        sys.stdout.flush()

        percent = (steps_completed / total_samples) * 100
        elapsed_str = format_time(int(elapsed))
        eta_str = format_time(int(eta_sec))

        header = f"{desc} | Step {steps_completed}/{total_samples} ({percent:.1f}%) | {iter_per_sec:.2f} it/s | Elapsed {elapsed_str} | ETA {eta_str}"
        loss_lines = [
            "  " + " | ".join(loss_items[i : i + 4])
            for i in range(0, len(loss_items), 4)
        ]
        print(header + "\nLosses:\n" + "\n".join(loss_lines))

    return format_progress_display


def create_training_loop(model: Any, dataset: Any) -> Callable:
    """Create training step function for one epoch.

    Args:
        model: Model with fit() method that returns dict with metrics.
        dataset: Training dataset with sample(), on_epoch_start(), on_epoch_end() methods.

    Returns:
        Callable that executes one training epoch with progress display.

    Raises:
        ValueError: If model or dataset is None or missing required methods.

    Example:
        >>> train_step = create_training_loop(model, dataset)
        >>> train_step("Epoch 1 / 10")  # Execute one training epoch
    """
    if model is None:
        raise ValueError("model cannot be None")
    if dataset is None:
        raise ValueError("dataset cannot be None")
    if not hasattr(model, "fit"):
        raise ValueError("model must have a fit() method")
    if not hasattr(dataset, "sample"):
        raise ValueError("dataset must have a sample() method")

    def training_step(desc: str) -> None:
        """Execute one training epoch with progress display.

        Args:
            desc: Description string for progress display (e.g., "Epoch 1 / 10").

        Side effects:
            Updates model with gradient steps and displays progress.
            Calls dataset lifecycle methods (on_epoch_start, on_epoch_end).
        """
        history = defaultdict(list)
        dataset.on_epoch_start()
        total_samples = len(dataset)
        format_progress = create_progress_display(desc, total_samples)

        for step in range(total_samples):
            sampled = dataset.sample()
            stats = model.fit(sampled)
            for k in stats.keys():
                history[k].append(stats[k])

            # Display summary every DISPLAY_FREQUENCY iterations
            if ((step + 1) % DISPLAY_FREQUENCY == 0) or step == 0:
                loss_items = [
                    f"{k}=%.4f" % np.mean(v) for k, v in sorted(history.items())
                ]
                format_progress(step, loss_items)

        format_progress(total_samples, loss_items)
        dataset.on_epoch_end()

    return training_step


def _pull_batches(batches, model):
    res = defaultdict(lambda: None)
    for batch in batches:
        res_batch = model.eval(batch)
        for k, v in res_batch.items():
            cur = res[k]
            if cur is not None:
                v = np.concatenate([cur, v], axis=0)
            res[k] = v
    return res


class EvaluationTracker:
    """Tracks best model performance across evaluation datasets.

    Encapsulates evaluation logic for gaze prediction tasks including loss tracking,
    model checkpointing, and early stopping.

    Attributes:
        best_loss: Overall best loss across all datasets (float).
        best_epoch: Epoch when best loss was achieved (int).
        losses: Per-dataset loss tracking (List[float]).
        last_output: Last evaluation output string (str).
        datasets: List of evaluation datasets.
        model: Model instance with eval() and save() methods.
        folder: Directory path for saving best model checkpoints.
        save_postfix: Postfix string for saving best model (default: "best").
    """

    def __init__(
        self,
        datasets: List[Any],
        model: Any,
        folder: Any = None,
        save_postfix: str = "best",
    ) -> None:
        """Initialize tracker with datasets and model.

        Args:
            datasets: List of test datasets.
            model: Model with eval() and save() methods.
            folder: Optional directory to save best model checkpoints.
            save_postfix: Postfix to use when saving best model (default: "best").

        Raises:
            ValueError: If model is None or missing required methods.
            TypeError: If datasets is not a list.
        """
        if model is None:
            raise ValueError("model cannot be None")
        if not hasattr(model, "eval"):
            raise ValueError("model must have an eval() method")
        if not isinstance(datasets, list):
            raise TypeError("datasets must be a list")
        if not datasets:
            raise ValueError("datasets list cannot be empty")

        self.datasets = datasets
        self.model = model
        self.folder = folder
        self.save_postfix = save_postfix
        self.best_loss: float = np.inf
        self.best_epoch: int = 0
        self.losses: List[float] = [np.inf] * len(datasets)
        self.last_output = ""

    def _eval_dataset(self, dataset: Any) -> Tuple[Dict[str, float], int]:
        """Evaluate model on single dataset.

        Returns metrics and sample count for weighted aggregation.

        Args:
            dataset: Test dataset to evaluate on.

        Returns:
            Tuple of (metrics_dict, sample_count).

        Example:
            >>> metrics, count = tracker._eval_dataset(test_dataset)
            >>> assert "total" in metrics
        """
        sample_count = 0

        def dataset_gen() -> Any:
            """Generate evaluation metrics for each batch in dataset.

            Yields model evaluation results for each batch.
            Updates nonlocal sample_count with total samples processed.
            """
            nonlocal sample_count
            dataset.reset()
            for batchId in range(len(dataset)):
                full = _pull_batches(dataset.sample(batchId), self.model)
                full["total"] = full["result"]
                sample_count += len(full["result"])
                yield full

        metrics = _mean_metrics(dataset_gen)
        return metrics, sample_count

    def evaluate_gaze(self, epoch: int = 0) -> bool:
        """Evaluate gaze prediction model on all datasets and track best model.

        Computes weighted average metrics across datasets using sample count weighting.
        Each dataset's metrics are weighted proportionally to its sample count,
        following standard ML evaluation practice (scikit-learn convention).

        Formula: weighted_metric = Σ(metric_i × count_i) / Σ(count_i)

        Updates last_output with evaluation metrics and epoch progress.

        Args:
            epoch: Current epoch number (for tracking and saving best model, default: 0).

        Returns:
            Weighted average loss across all datasets (float).

        Raises:
            ValueError: If no datasets or no metrics collected.

        Example:
            >>> loss = tracker.evaluate_gaze(epoch=5)
            >>> if loss < tracker.best_loss:
            ...     print("New best model!")
        """
        # Collect metrics and sample counts from each dataset
        dataset_metrics = []
        dataset_sample_counts = []

        for dataset in self.datasets:
            metrics, sample_count = self._eval_dataset(dataset)
            dataset_metrics.append(metrics)
            dataset_sample_counts.append(sample_count)

        # Validate we have metrics
        if not dataset_metrics:
            raise ValueError("No metrics collected from any dataset")

        # Collect all metric keys from all datasets (union of all keys)
        all_metric_keys: Set[str] = set()
        for metrics in dataset_metrics:
            all_metric_keys.update(metrics.keys())

        if not all_metric_keys:
            raise ValueError("No metric keys found in any dataset")

        # Compute total samples for normalization
        total_samples = sum(dataset_sample_counts)
        if total_samples <= 0:
            raise ValueError(f"All datasets have zero samples: {dataset_sample_counts}")

        all_metrics: Dict[str, float] = {}

        # Compute weighted average: Σ(metric_i × (count_i / count))
        for key in all_metric_keys:
            all_metrics[key] = sum(
                metrics.get(key, 0.0) * (count / float(total_samples))
                for metrics, count in zip(dataset_metrics, dataset_sample_counts)
            )

        mean_loss = all_metrics.get("total", np.inf)

        is_better = mean_loss < self.best_loss
        # Track best model
        if is_better:
            self.best_loss = mean_loss
            self.best_epoch = epoch
            self.model.save(str(self.folder), postfix=self.save_postfix)

        # Build output with metrics and progress
        metrics_parts = [f"{k}: {v:.5f}" for k, v in all_metrics.items()]
        metrics_str = " | ".join(metrics_parts)
        progress = f"Passed {epoch - self.best_epoch} epochs since the last improvement (best: {self.best_loss:.5f})"
        self.last_output = f"{metrics_str}\n{progress}"

        return is_better


def format_epoch_desc(epoch: int, total_epochs: int) -> str:
    """Format epoch description string with proper padding.

    Args:
        epoch: Current epoch (1-indexed).
        total_epochs: Total number of epochs.

    Returns:
        Formatted string like "Epoch 001 / 100" (with padding to match total).

    Example:
        >>> desc = format_epoch_desc(5, 100)
        >>> assert desc == "Epoch 005 / 100"
    """
    num_digits = len(str(total_epochs))
    return f"Epoch {epoch:0{num_digits}d} / {total_epochs}"
