#!/usr/bin/env python3

"""
Show samples where the "full" variant (face mesh + eyes) underperforms
compared to "no_face" (eyes only) or "no_eyes" (face mesh only) variants.

This script analyzes model performance on individual samples and displays
only the last frame of samples where the full variant is not optimal,
grouped by 4 samples at a time.
"""

import argparse
import logging
import numpy as np
import cv2
from typing import Dict, List
from Core.data.sample_viewer import create_visualization
from collections import defaultdict
from pathlib import Path
from Core import Utils

from Core.models import ModelWrapper
from Core.data.TestLoader import TestLoader

logger = logging.getLogger(__name__)
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
    else:
        return {"model": weights_str, "postfix": "best"}


def _model_from(args: argparse.Namespace, stats: dict, folder: Path) -> ModelWrapper:
    wrapper_args = dict(timesteps=args.steps, stats=stats)
    wrapper_args = {
        **wrapper_args,
        "scale_mult": args.model_scale,
    }

    weight_info = _parse_model_weights(args.model)
    wrapper_args["weights"] = dict(
        folder=str(folder),
        postfix=weight_info["postfix"],
        force=False,
    )
    wrapper_args["model"] = weight_info["model"]
    return ModelWrapper(**wrapper_args)


def load_test_loaders(dataset_path: str) -> list:
    """Load test loaders using the same pattern as train.py."""
    folder = Path(dataset_path)
    # find folders with the same pattern as train.py: "test-main/test-*/"
    test_folders = [str(nm) for nm in sorted(folder.glob("test-main/test-*/"))]
    test_loaders = [TestLoader(nm) for nm in test_folders]
    test_loaders = [ds for ds in test_loaders if 0 < len(ds)]
    return test_loaders


def evaluate_batch(
    model: ModelWrapper, loader: TestLoader, batch_idx: int
) -> Dict[str, float]:
    """Evaluate a single batch under all three modalities."""

    # Get full performance (face mesh + eyes)
    def eval(no_face=False, no_eyes=False):
        batch = loader.sample(batch_idx, no_face=no_face, no_eyes=no_eyes)
        return model.eval(batch)["result"], batch

    full, batch = eval()
    no_face, _ = eval(no_face=True, no_eyes=False)
    no_eyes, _ = eval(no_face=False, no_eyes=True)

    best = np.minimum(np.minimum(no_eyes, no_face), full)
    not_best_batch, _ = np.where(full != best)
    batch = Utils.to_numpy(batch[0])
    return {k: v[not_best_batch, -1:] for k, v in batch.items()}


def filtered_samples(
    model: ModelWrapper, test_loaders: List[TestLoader], batch_size: int = 4
):
    """
    Generator that yields batches of 4 samples where the 'full' variant underperforms.

    Args:
        model: Loaded model for evaluation
        test_loaders: List of test loaders
        batch_size: Number of samples to group together (default: 4)
        start_batch: Starting batch ID to begin evaluation from

    Yields:
        List of sample dictionaries
    """
    current_batch = defaultdict(list)

    def serve(force=False):
        nonlocal current_batch
        if not current_batch:
            return
        k = list(current_batch.keys())[0]
        N = len(current_batch[k])

        for idx in range(0, N, batch_size):
            res = {k: v[idx : idx + batch_size] for k, v in current_batch.items()}
            if batch_size <= len(res[k]) or force:
                yield res

        current_batch = {k: v[idx:] for k, v in current_batch.items()}

    for loader_idx, loader in enumerate(test_loaders):
        logger.info(
            f"Processing loader {loader_idx + 1}/{len(test_loaders)} ({len(loader)} batches)"
        )

        for batch_idx in range(len(loader)):
            # Evaluate this batch under different modalities
            samples = evaluate_batch(model, loader, batch_idx)
            for k, v in samples.items():
                if 0 < len(v):
                    if 0 < len(current_batch[k]):
                        v = np.concatenate([current_batch[k], v], axis=0)
                    current_batch[k] = v
            yield from serve()

    # Yield any remaining samples
    yield from serve(force=True)


def main(args):
    """Main entry point for the full-not-best samples viewer."""
    KEY_ESC = 27  # Escape key
    KEY_SPACE = 32  # Space key

    folder = Path(args.folder) / "Data"
    stats = Utils.read_json(str(folder / "remote" / "stats.json"))

    # Load model and test loaders
    model = _model_from(args, stats, folder)
    test_loaders = load_test_loaders(folder)

    # Find and show underperforming samples in batches
    for sample_batch in filtered_samples(model, test_loaders, args.batch_size):
        img = create_visualization(
            {"clean": sample_batch, "augmented": sample_batch}, zoom_factor=4
        )
        # Display
        cv2.imshow(
            winname="Eye Samples: Clean vs Augmented (ESC=exit, SPACE=next, other=prev)",
            mat=img,
        )

        # Wait for key press with longer timeout for viewing
        key = cv2.waitKey(delay=0) & 0xFF

        if key == KEY_ESC:
            logger.info("Exiting.")
            break
        elif key == KEY_SPACE:
            # Continue to next sample
            continue
        else:
            # Any other key: show previous sample again
            if img is not None:
                cv2.imshow(
                    winname="Eye Samples: Clean vs Augmented (ESC=exit, SPACE=next, other=prev)",
                    mat=img,
                )
                cv2.waitKey(delay=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show samples where the "full" variant underperforms compared to other variants',
    )

    parser.add_argument(
        "--model", required=True, help="model (model/{checkpoint or best})"
    )

    parser.add_argument(
        "--model-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for the model (default: 1.0)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument("--folder", type=str, default=str(ROOT_FOLDER))
    main(parser.parse_args())
