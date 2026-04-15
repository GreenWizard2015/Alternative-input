#!/usr/bin/env python3
"""
Data Filtering Script

Simplified script that loads sample filters from sample_filters.json and processes
datasets in the Data/remote/ directory to generate filtered training and testing datasets.

Usage:
    python scripts/prepare-filter.py --test-fraction 0.2 --random-seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Core.logging_config import get_logger


# Configuration
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_RANDOM_SEED = 42

# File paths
REMOTE_DATA_DIR = Path("Data/remote")
SAMPLE_FILTERS_FILE = Path("sample_filters.json")
OUTPUT_TRAIN_FILE = Path("Data/filter-train.npz")
OUTPUT_TEST_FILE = Path("Data/filter-test.npz")


def scan_datasets(remote_dir: Path) -> List[Path]:
    """Find all top-level directories containing .npz files in Data/remote/."""
    logger = get_logger("DatasetScanner")
    logger.info(f"Scanning for datasets in {remote_dir}")

    if not remote_dir.exists():
        raise FileNotFoundError(f"Remote data directory not found: {remote_dir}")

    datasets = []
    for all_npz in remote_dir.rglob("all.npz"):
        datasets.append(all_npz.parent)

    if not datasets:
        raise FileNotFoundError(f"No datasets found in {remote_dir}")

    logger.info(f"Found {len(datasets)} datasets")
    return datasets


def load_filters(filter_file: Path) -> Dict[str, Dict[str, int]]:
    """Load and parse sample_filters.json."""
    logger = get_logger("SampleFilterLoader")
    logger.info(f"Loading filters from {filter_file}")

    if not filter_file.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_file}")

    with open(filter_file, "r") as f:
        filters = json.load(f)
    logger.info(f"Loaded filters for {len(filters)} datasets")
    return filters


def get_filter_flags(
    filters: Dict[str, Dict[str, int]], dataset_name: str
) -> Dict[int, bool]:
    """Get boolean array indicating valid samples for a dataset."""
    # Find the matching filter key
    matching_key = None
    for key in filters.keys():
        if dataset_name in key:
            matching_key = key
            break

    if matching_key is None:
        return None

    dataset_filters = filters[matching_key]
    res = {int(k): v for k, v in dataset_filters.items()}
    return res


def load_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and filter dataset from .npz file."""
    logger = get_logger("NPZLoader")

    # Find all.npz file
    npz_file = next(dataset_path.rglob("all.npz"), None)
    if not npz_file:
        raise FileNotFoundError(f"all.npz not found in dataset: {dataset_path}")

    logger.info(f"Loading samples from {npz_file}")
    data = np.load(npz_file, allow_pickle=True)

    # Validate required keys
    required_keys = ["left eye", "right eye"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in .npz file: {missing_keys}")

    X_left = data["left eye"]
    X_right = data["right eye"]

    # Validate shapes
    if X_left.shape != X_right.shape:
        raise ValueError(
            f"Shape mismatch between left and right eye images: {X_left.shape} vs {X_right.shape}"
        )

    return X_left, X_right


def apply_filters(
    X_left: np.ndarray, X_right: np.ndarray, filter_flags: Dict[int, bool]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply filter flags to dataset samples."""
    keys = sorted(filter_flags.keys())
    valid_mask = np.zeros((len(X_left)), bool)
    for k in keys:
        valid_mask[k] = True
    filtered_X_left = X_left[valid_mask]
    filtered_X_right = X_right[valid_mask]

    return filtered_X_left, filtered_X_right, np.array([filter_flags[k] for k in keys])


def split_indices(
    labels: np.array, test_fraction: float, random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split sample indices into train and test sets."""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    test_indices = []
    train_indices = []
    by_label = {True: [], False: []}
    for idx, lbl in enumerate(labels):
        by_label[lbl].append(idx)

    by_label = {k: np.array(v) for k, v in by_label.items()}
    for k, v in by_label.items():
        np.random.shuffle(v)
        num_test = int(len(v) * test_fraction)
        test_indices.append(v[:num_test])
        train_indices.append(v[num_test:])

    return (
        np.concatenate(test_indices, axis=0).astype(np.int32),
        np.concatenate(train_indices, axis=0).astype(np.int32),
    )


def write_output_file(
    file_path: Path, X_left: np.ndarray, X_right: np.ndarray, is_valid: np.ndarray
) -> None:
    """Write filtered data to output file."""
    logger = get_logger("OutputWriter")
    logger.info(f"Writing data to {file_path}")

    output_data = {"left eye": X_left, "right eye": X_right, "is_valid": is_valid}

    np.savez_compressed(file_path, **output_data)
    logger.info(f"Data saved: {len(is_valid)} samples")


def process_datasets(
    test_fraction: float = DEFAULT_TEST_FRACTION, random_seed: int = DEFAULT_RANDOM_SEED
) -> None:
    """Process all datasets and generate filtered output files."""
    logger = get_logger("FilterProcessor")
    logger.info("Starting data filtering process")

    # Load filters
    filters = load_filters(SAMPLE_FILTERS_FILE)

    # Scan datasets
    datasets = scan_datasets(REMOTE_DATA_DIR)

    # Process each dataset
    all_train_X_left, all_train_X_right, all_train_is_valid = [], [], []
    all_test_X_left, all_test_X_right, all_test_is_valid = [], [], []

    for dataset_path in datasets:
        dataset_name = str(dataset_path).replace(str(REMOTE_DATA_DIR), "").strip("/")

        # Get filter flags and apply filters
        filter_flags = get_filter_flags(filters, dataset_name)
        if filter_flags is None:
            continue
        # Load dataset
        X_left, X_right = load_dataset(dataset_path)
        filtered_X_left, filtered_X_right, is_valid = apply_filters(
            X_left, X_right, filter_flags
        )

        if len(is_valid) == 0:
            logger.info(f"Dataset {dataset_name}: no valid samples found")
            continue

        # Split filtered dataset
        test_indices, train_indices = split_indices(
            is_valid, test_fraction, random_seed
        )

        # Add to train/test arrays
        all_train_X_left.append(filtered_X_left[train_indices])
        all_train_X_right.append(filtered_X_right[train_indices])
        all_train_is_valid.append(is_valid[train_indices])

        all_test_X_left.append(filtered_X_left[test_indices])
        all_test_X_right.append(filtered_X_right[test_indices])
        all_test_is_valid.append(is_valid[test_indices])

        logger.info(
            f"Dataset {dataset_name}: {len(train_indices)} train, {len(test_indices)} test samples"
        )

    # Combine and save datasets
    logger.info("Combining train datasets")
    combined_train = (
        np.concatenate(all_train_X_left, axis=0),
        np.concatenate(all_train_X_right, axis=0),
        np.concatenate(all_train_is_valid, axis=0),
    )

    logger.info("Combining test datasets")
    combined_test = (
        np.concatenate(all_test_X_left, axis=0),
        np.concatenate(all_test_X_right, axis=0),
        np.concatenate(all_test_is_valid, axis=0),
    )

    # Save output
    write_output_file(OUTPUT_TRAIN_FILE, *combined_train)
    write_output_file(OUTPUT_TEST_FILE, *combined_test)

    total_train, total_test = len(combined_train[2]), len(combined_test[2])
    logger.info(f"Completed: {total_train} train, {total_test} test samples")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare filtered datasets for training"
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of samples to use for testing (default: 0.2)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible splits (default: 42)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate arguments
    if not 0.0 <= args.test_fraction <= 1.0:
        raise ValueError("test-fraction must be between 0.0 and 1.0")

    logger = get_logger("prepare-filter")
    logger.info(
        f"Starting with test_fraction={args.test_fraction}, seed={args.random_seed}"
    )

    process_datasets(test_fraction=args.test_fraction, random_seed=args.random_seed)
    logger.info("Data filtering completed successfully!")


if __name__ == "__main__":
    main()
