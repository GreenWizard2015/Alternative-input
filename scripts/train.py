#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training script for gaze prediction models.

Orchestrates the full training pipeline including:
- Dataset loading and sampling (using temporal sequences)
- Model initialization and training loop
- Evaluation on test set
- Checkpoint management and metrics tracking

Usage:
    python train.py --folder /path/to/dataset --batch-size 32 --steps 4

The script loads data from a structured folder with remote dataset and stats,
samples temporal sequences with augmentation, and trains the model with periodic
evaluation on a test set.
"""

from typing import List
from pathlib import Path
import argparse
from Core.models import ModelStudentTrainer, ModelWrapper
import Core.Utils as Utils
from Core.data.TestLoader import TestLoader
from Core.data.DataSampler import DataSampler
from Core.data.DatasetLoader import DatasetLoader
from Core.data.AugmentationDefaults import DEFAULT_AUGMENTATION_PARAMS
from scripts.training_utils import (
    create_training_loop,
    EvaluationTracker,
    format_epoch_desc,
)
import numpy as np
import tensorflow as tf
from Core.logging_config import get_logger

logger = get_logger(__name__)
ROOT_FOLDER = Path(__file__).parent.parent


def _parse_model_weights(weights_str: str) -> dict:
    """Parse weights string in format 'model/postfix' or just 'postfix'.

    Args:
        weights_str: Weight specification string

    Returns:
        Dictionary with model and postfix keys
    """
    if "/" in weights_str:
        model_name, postfix = weights_str.split("/", 1)
        return {"model": model_name, "postfix": postfix}
    return {"model": weights_str, "postfix": "best"}


def _transfer_model_weights(teacher, student):
    def assign(w1, w2):
        # reshape w1 to match w2 and assign
        shape1 = w1.shape
        if 0 == len(shape1):
            shape1 = (1,)
        shape2 = w2.shape
        if 0 == len(shape2):
            shape2 = (1,)
        total1 = np.prod(shape1)
        total2 = np.prod(shape2)
        flat = (1, -1)
        w1 = tf.reshape(w1, flat)
        w1 = tf.tile(w1, (int(1 + total2 // total1), 1))
        w1 = tf.reshape(w1, (-1,))
        w1 = tf.reshape(w1[:total2], w2.shape)
        w2.assign(w1)

    teacher_weights = teacher.trainable_variables
    student_weights = student.trainable_variables
    assert len(teacher_weights) == len(student_weights)
    for w1, w2 in zip(teacher_weights, student_weights):
        assign(w1, w2)


def _create_wrapper(
    args, folder, scale, model_prefix, mode, weights, embeddings, force, cache_id
):
    wrapper_args = {
        **args,
        "scale_mult": scale,
        "model": f"{model_prefix}-{scale:.1f}",
        "mode": mode,
        "cache_id": cache_id,
    }
    # Load weights if provided
    weights_res = None
    if weights is not None:
        weight_info = _parse_model_weights(weights)
        wrapper_args["weights"] = weights_res = dict(
            folder=str(folder),
            postfix=weight_info["postfix"],
            embeddings=embeddings,
            force=force,
        )
        if "model" in weight_info:
            wrapper_args["model"] = weight_info["model"]

    return ModelWrapper(**wrapper_args), weights_res


def _parse_teacher_clones(teacher_clones: str, teacher_count: int) -> List[int]:
    """Parse teacher clones string into list of clone counts per teacher.

    Args:
        teacher_clones: String with clone counts, can be single number or comma-separated list

    Returns:
        List of clone counts for each teacher
    """
    if not teacher_clones:
        teacher_clones = "1"

    # Parse single number or comma-separated list
    clone_parts = [int(x.strip()) for x in teacher_clones.split(",")]
    if len(clone_parts) == 1:
        # Single number applies to all teachers
        return clone_parts * teacher_count
    return clone_parts


def _teachers_from(
    args: argparse.Namespace, stats: dict, folder: Path
) -> List[ModelWrapper]:
    wrapper_args = dict(timesteps=args.steps, stats=stats)
    teacher_scales = list(args.teacher_scale.split(","))
    teacher_weights = list(args.teacher_weights.split(","))
    assert len(teacher_scales) == len(teacher_weights)

    # Parse teacher clones
    teacher_clones = _parse_teacher_clones(args.teacher_clones, len(teacher_weights))
    assert len(teacher_clones) == len(teacher_weights)

    def model_args(cache_id, scale, weights, model_prefix="teacher"):
        return dict(
            args=wrapper_args,
            folder=str(folder),
            model_prefix=model_prefix,
            mode="full",
            embeddings=True,
            force=args.force,
            cache_id=cache_id,
            scale=float(scale),
            weights=weights,
        )

    wrappers = [[] for _ in range(max(teacher_clones))]
    for idx, (weight, scale, clones) in enumerate(
        zip(teacher_weights, teacher_scales, teacher_clones)
    ):
        model_params = model_args(
            cache_id=f"teacher-{idx}", scale=scale, weights=weight
        )
        for clone_idx in range(clones):
            teacher_wrapper, _ = _create_wrapper(**model_params)
            wrappers[clone_idx].append(teacher_wrapper)

    res = []
    for lst in wrappers:
        res.extend(lst)
    return res


def _student_from(args: argparse.Namespace, stats: dict, folder: Path) -> ModelWrapper:
    wrapper_args = dict(timesteps=args.steps, stats=stats)
    student_wrapper, student_weights = _create_wrapper(
        wrapper_args,
        folder=str(folder),
        scale=args.student_scale,
        model_prefix="student",
        mode=args.mode,
        weights=args.student_weights,
        embeddings=not args.no_embeddings,
        force=args.force,
    )

    return student_wrapper, student_weights


def _trainer_from(
    args: argparse.Namespace, stats: dict, folder: Path
) -> ModelStudentTrainer:
    """Instantiate trainer with model wrapper based on command-line arguments.

    Creates the appropriate trainer (Student or Teacher) with:
    - ModelWrapper for student model
    - Optional teacher model for knowledge distillation
    - Pre-trained weights if provided

    Args:
        args: Parsed command-line arguments containing trainer name and model config
        stats: Dataset statistics dictionary
        folder: Data folder path for loading/saving weights

    Returns:
        Instantiated trainer

    Raises:
        ValueError: If requested trainer type is unknown.

    Example:
        >>> trainer = _trainer_from(args, stats, folder=Path("data"))
        >>> trainer.summary()
    """
    teacher_wrappers = _teachers_from(args, stats, folder)
    student_weights = None
    if args.student_index is not None:
        student_wrapper = teacher_wrappers.pop(args.student_index)
    else:
        student_wrapper, student_weights = _student_from(
            args, stats=stats, folder=folder
        )
        if student_weights is None and (0 < len(teacher_wrappers)):
            _transfer_model_weights(teacher_wrappers[0], student_wrapper)

    if args.no_embeddings:
        student_wrapper.reset_embeddings()

    exclude = []
    if args.exclude:
        assert args.exclude in ["final", "intermediate"]
        exclude = [args.exclude]

    return ModelStudentTrainer(
        model_wrapper=student_wrapper,
        teachers_models=teacher_wrappers,
        feature_match_loss_weight=args.feature_match_weight,
        weights=student_weights,
        exclude=exclude,
    )


def main(args: argparse.Namespace) -> None:
    """Execute the training pipeline.

    Loads datasets, initializes model, and runs training loop with periodic
    evaluation. Uses early stopping based on validation performance.

    Args:
        args: Command-line arguments including:
            - folder: Path to dataset folder
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - batch_per_epoch: Number of samples per epoch (default: 20000)
            - patience: Early stopping patience in epochs
            - steps: Number of timesteps per sample
            - sampling: Sampling strategy
            - model: Optional pre-trained model to load
            - embeddings: Optional embedding weights to load
            - mode: Training mode "full" (two-stage, default) or "encoder" (Face2Step only)
            - teacher_scale: Teacher model scale multiplier for knowledge distillation (default: 1.0)
            - teacher_weights: Path to pre-trained teacher weights (optional)
            - feature_match_weight: Feature matching loss weight [0.0-1.0] (default: 0.5)
            - adapter_intermediate_dim: Adapter bottleneck dimension (optional, default: geometric mean)

    Returns:
        None (saves best and latest model checkpoints to disk)
    """
    timesteps = args.steps
    folder = Path(args.folder) / "Data"
    json_path = str(folder / "remote" / "stats.json")
    stats = Utils.read_json(json_path)

    trainDataset = DatasetLoader(
        json_path,
        samplerArgs=dict(
            batch_size=args.batch_size,
            minFrames=timesteps,
            maxT=1.0,
            defaults=dict(
                timesteps=timesteps,
                stepsSampling="uniform",
                **DEFAULT_AUGMENTATION_PARAMS,
            ),
        ),
        sampler_class=DataSampler,
        batchPerEpoch=args.batch_per_epoch,
        sampling_mode=args.sampling,
    )
    # Instantiate trainer with model wrapper
    model = _trainer_from(args, stats, folder)

    # find folders with the name "/test-*/"
    evalDatasets = [
        TestLoader(str(nm), batch_size=args.test_batch_size)
        for nm in sorted(folder.glob("test-main/test-*/"))
    ]
    evalDatasets = [ds for ds in evalDatasets if 0 < len(ds)]
    # Use EvaluationTracker for best model management
    tracker = EvaluationTracker(evalDatasets, model, folder=folder, save_postfix="best")
    tracker.evaluate_gaze(epoch=0)  # evaluate loaded model

    # Create training loop using utilities
    trainStep = create_training_loop(model, trainDataset)
    for epoch in range(1, args.epochs + 1):
        trainStep(
            desc=f"{tracker.last_output}\n\n{format_epoch_desc(epoch, args.epochs)}"
        )
        model.save(str(folder), postfix="latest")
        tracker.evaluate_gaze(epoch=epoch)  # evaluate loaded model
        logger.info(tracker.last_output)
        if args.patience <= (epoch - tracker.best_epoch):
            logger.info("Early stopping")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-per-epoch", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-embeddings", default=False, action="store_true")
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Force training to continue even if weights are missing.",
    )
    parser.add_argument("--folder", type=str, default=str(ROOT_FOLDER))
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "encoder"],
        help="Training mode: 'full' (two-stage with temporal) or 'encoder' (Face2Step only). Default: full",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="oversample",
        choices=["oversample", "undersample"],
        help="Dataset sampling strategy",
    )

    # Knowledge Distillation Arguments
    parser.add_argument(
        "--teacher-scale",
        type=str,
        default="",
        help="Teacher model scale multiplier.",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default="",
        help="Path to pre-trained teacher model weights (optional). "
        "Format: 'postfix' or 'model/postfix'. "
        "If provided, loads weights into frozen teacher for distillation. "
        "Weights must match --teacher-scale dimensions.",
    )
    parser.add_argument(
        "--teacher-clones",
        type=str,
        default="",
        help="How many times clone pre-trained teacher model. Can be one number for all or comma-separated list.",
    )
    parser.add_argument(
        "--student-index",
        type=int,
        default=None,
        help="Index of teacher to become a student model (optional). ",
    )
    parser.add_argument(
        "--student-weights",
        type=str,
        default=None,
        help="Path to pre-trained student model weights (optional). ",
    )
    parser.add_argument(
        "--student-scale",
        type=float,
        default=1.0,
        help="Student model scale multiplier.",
    )
    parser.add_argument(
        "--feature-match-weight",
        type=float,
        default=1.0,
        help="Feature matching loss weight [0.0-1.0] (default: 1.0). "
        "Controls balance between task loss and latent feature matching. ",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Test batch size for sub-batching. If None, use full npz batch sizes. "
        "Useful for memory management or performance optimization.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Excluded from distillation parts. Can be 'final' or 'intermediate'.",
    )
    args = parser.parse_args()
    main(args)
