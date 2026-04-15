#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training script for filter classification models.

Orchestrates the complete training pipeline for binary classification of eye image validity:
- NPZ data loading with custom FilterDataLoader
- FilterModel initialization and training loop
- Evaluation on test set using EvaluationTracker
- Model checkpointing and metrics tracking

Usage:
    python train-filter.py --data-folder /path/to/data --epochs 100 --batch-size 32

The script loads filter data from NPZ files, trains the binary classification model,
and saves results in NPZ format for easy loading and deployment.
"""

from pathlib import Path
import argparse
from Core.data.FilterDataLoader import FilterDataLoader
from Core.models import FilterWrapper
from scripts.training_utils import (
    create_training_loop,
    EvaluationTracker,
    format_epoch_desc,
)
from Core.logging_config import get_logger

logger = get_logger(__name__)
ROOT_FOLDER = Path(__file__).parent.parent


def main(args: argparse.Namespace) -> None:
    """Execute the filter training pipeline.

    Loads filter datasets, initializes model, and runs training loop with periodic
    evaluation. Uses early stopping based on validation performance.

    Args:
        args: Command-line arguments including:
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - batch_per_epoch: Number of samples per epoch (default: -1 for full dataset)
            - patience: Early stopping patience in epochs
            - folder: Base folder path for data
            - data_folder: Path to data folder with filter-train.npz and filter-test.npz
            - learning_rate: Learning rate for optimizer
            - save_folder: Path to save model checkpoints
            - weights: Optional pre-trained model weights to load
            - force: Force training to continue even if weights are missing

    Returns:
        None (saves best and latest model checkpoints to disk)
    """
    # Setup paths
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    train_file = data_folder / "filter-train.npz"
    test_file = data_folder / "filter-test.npz"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    logger.info("Starting filter training...")
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Training file: {train_file}")
    logger.info(f"Test file: {test_file}")

    # Create data loaders with specific NPZ file paths
    train_loader = FilterDataLoader(
        npz_file=str(train_file),
        batch_size=args.batch_size,
        train=True,
        batch_per_epoch=args.batch_per_epoch,
    )
    test_loader = FilterDataLoader(npz_file=str(test_file), batch_size=args.batch_size)
    logger.info(f"Training samples: {len(train_loader) * args.batch_size}")
    logger.info(f"Test samples: {len(test_loader) * args.batch_size}")

    save_folder = Path(args.save_folder) if args.save_folder else data_folder / "models"
    # Create FilterWrapper with proper weight loading
    model_wrapper = FilterWrapper(
        model="filter",
        weights=(
            None
            if args.weights is None
            else dict(
                folder=str(save_folder),
                postfix="best",
                force=args.force,
            )
        ),
    )

    logger.info("Model compiled successfully")
    # model_wrapper.summary()

    # Load weights if provided
    if args.weights is not None:
        logger.info(f"Loaded weights from {args.weights}")

    # Setup evaluation datasets for EvaluationTracker
    evalDatasets = [test_loader]

    # Use EvaluationTracker for best model management
    tracker = EvaluationTracker(
        evalDatasets, model_wrapper, folder=save_folder, save_postfix="best"
    )
    tracker.evaluate_gaze(epoch=0)  # evaluate loaded model

    # Create training loop using utilities
    trainStep = create_training_loop(model_wrapper, train_loader)
    for epoch in range(1, args.epochs + 1):
        trainStep(
            desc=f"{tracker.last_output}\n\n{format_epoch_desc(epoch, args.epochs)}"
        )
        model_wrapper.save(str(save_folder), postfix="latest")
        tracker.evaluate_gaze(epoch=epoch)  # evaluate loaded model
        logger.info(tracker.last_output)
        if args.patience <= (epoch - tracker.best_epoch):
            logger.info("Early stopping")
            break

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-per-epoch", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Force training to continue even if weights are missing.",
    )
    parser.add_argument("--folder", type=str, default=str(ROOT_FOLDER))
    parser.add_argument(
        "--data-folder",
        type=str,
        default=str(ROOT_FOLDER / "Data"),
        help="Path to data folder with filter-train.npz and filter-test.npz",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--save-folder", type=str, default=None, help="Path to save model checkpoints"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained model weights (optional). "
        "Format: 'postfix' or 'model/postfix'.",
    )

    args = parser.parse_args()
    main(args)
