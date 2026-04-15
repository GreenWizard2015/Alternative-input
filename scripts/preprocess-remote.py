#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample-based preprocessing with trajectory validation.

This script performs the following steps:
  1. Load the npz files recursively from the "Data/remote" folder
  2. Combine the npz files into a single dataset
  3. Drop the "userId", "screenId", "cameraId", and "placeId" fields (monitorId preserved)
  4. Identify valid samples using FilteredDataset with trajectory validation
  5. Randomly split valid samples into training and testing sets
  6. Extract frame unions for each split using used_samples property
  7. Save the training.npz and testing.npz files with guaranteed valid trajectories
  8. Remove the npz files from the current folder

Key improvements over frame-based splitting:
  - Sample-level split ensures principled separation
  - Trajectory validation guarantees all training samples have valid context
  - Frame-level union prevents data leakage between train/test splits
  - Reproducible splits via random seed

Also generates "Data/remote/stats.json" file with the following structure:
  {
    "userId": [ ... ],
    "screenId": [ ... ],
    "cameraId": [ ... ],
    "monitorId": [ ... ],
    "placeId": [ ... ],
    "blacklist": [[userId, screenId, cameraId, monitorId, placeId], ...],
  }
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, List, Any, Union, Callable, Optional
from functools import lru_cache
import numpy as np
import argparse
import Core.Utils as Utils
from Core.Utils import DatasetPath
from Core.logging_config import get_logger
from scripts.visualization_utils import plot_histogram
from scripts.Constants import MAX_DELTA_THRESHOLD
from scripts.preprocessing.core import (
    remove_frames_with_zero_time_delta,
    expand_indices_for_trajectories,
    validate_dataset_sparsity,
)
from Core.Constants import HIERARCHY_LEVELS, ID_MONITOR
from Core.models import FilterWrapper

logger = get_logger(__name__)
ROOT_FOLDER = Path(__file__).parent.parent

# Constants for filtering
DEFAULT_FILTER_BATCH_SIZE = 512
DEFAULT_FILTER_THRESHOLD = None  # disabled by default


def loadNpz(path: str) -> Dict[str, np.ndarray]:
    """Load and validate dataset from npz file.

    Args:
        path: Path to npz file or directory.

    Returns:
        Dictionary of numpy arrays with userId, screenId, cameraId, and placeId removed (monitorId preserved).

    Raises:
        AssertionError: If dataset contains multiple unique values for any ID field.
        FileNotFoundError: If file does not exist.

    Example:
        >>> data = loadNpz("data/remote/userId/screenId/cameraId/monitorId/placeId/")
        >>> assert "monitorId" in data
    """
    res = Utils.dataset_from(path)
    if res is None:
        return {}
    # validate the dataset and remove ID fields except monitorId
    # monitorId is preserved as it's needed for dataset path organization
    fields_to_remove = [id_key for id_key in HIERARCHY_LEVELS if id_key != ID_MONITOR]
    for field_name in fields_to_remove:
        if field_name in res:
            v = res.pop(field_name)
            v = np.unique(v)
            assert 1 == len(
                v
            ), f"Expecting single {field_name}, got {len(v)} unique values: {v}"

    return res


def dropZeroTimeDelta(dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove frames with zero or negative time deltas.

    Uses the refactored preprocessing module for consistency.

    Args:
        dataset: Dictionary of numpy arrays with "time" key containing timestamps.

    Returns:
        Dictionary with frames filtered to keep only those with positive time deltas.

    Example:
        >>> filtered = dropZeroTimeDelta(dataset)
        >>> assert np.all(np.diff(filtered["time"]) > 0)
    """
    return remove_frames_with_zero_time_delta(dataset)


def expand_indices(indices: np.ndarray, minimumFrames: int) -> np.ndarray:
    """Expand sample indices to frame window indices.

    Uses the refactored preprocessing module for consistency.

    Args:
        indices: Sample indices to expand.
        minimumFrames: Minimum number of frames per trajectory.

    Returns:
        Unique sorted array of frame indices covering all trajectories.

    Example:
        >>> expanded = expand_indices(np.array([5, 10]), 3)
        >>> assert len(expanded) >= 6  # Each sample expands to minimumFrames indices
    """
    return expand_indices_for_trajectories(
        np.array(indices, dtype=np.int64), minimumFrames
    )


def create_filtered_dataset(
    dataset: Dict[str, np.ndarray],
    sample_indices: np.ndarray,
    minimumFrames: int,
    maxT: float,
):
    """Create a FilteredDataset with given samples.

    Args:
        dataset: Dictionary of numpy arrays containing the full dataset.
        sample_indices: Indices of samples to include in this split.
        minimumFrames: Minimum number of frames required for trajectory validation.
        maxT: Maximum time window for sample trajectory validation (seconds).

    Returns:
        FilteredDataset containing the specified samples.

    Raises:
        ValueError: If sample_indices are invalid.
    """
    from Core.data.FilteredDataset import FilteredDataset
    from Core.data.SamplesStorage import SamplesStorage

    storage = SamplesStorage(
        userId=-1, screenId=-1, cameraId=-1, monitorId=-1, placeId=-1
    )
    filtered = FilteredDataset(storage=storage, minFrames=minimumFrames, maxT=maxT)

    # Add samples (in sorted order to maintain temporal continuity)
    filtered.add_samples_block({k: v[sample_indices] for k, v in dataset.items()})
    return filtered


# ============================================================================
# FILTER WRAPPER INTEGRATION FUNCTIONS
# ============================================================================


@lru_cache(maxsize=1)
def load_filter_model() -> Optional[FilterWrapper]:
    """Load FilterWrapper model with graceful error handling.
    Uses LRU caching to prevent TensorFlow session conflicts during batch processing.

    Returns:
        FilterWrapper model instance if successful, None if loading fails.
    """
    # Create FilterWrapper and load model
    filter_model = FilterWrapper(model="filter")
    load_folder = str(Path(__file__).parent.parent / "Data" / "models")
    logger.info(f"Loading from folder: {load_folder} with postfix: best")
    filter_model.load(folder=load_folder, postfix="best")
    logger.info("FilterWrapper model loaded successfully")
    return filter_model


def prepare_filter_batch(
    dataset: Dict[str, np.ndarray], sample_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """Prepare batch data for FilterWrapper from dataset.

    Args:
        dataset: Complete dataset dictionary.
        sample_indices: Array of sample indices to include in batch.

    Returns:
        Dictionary with "left eye" and "right eye" arrays ready for filtering.
    """
    # Check required keys exist
    if "left eye" not in dataset or "right eye" not in dataset:
        raise ValueError("Dataset missing required eye data keys")

    # Extract eye images for batch
    left_eyes = dataset["left eye"][sample_indices]  # (batch, 48, 48)
    right_eyes = dataset["right eye"][sample_indices]  # (batch, 48, 48)

    # Validate shapes
    if len(left_eyes) != len(sample_indices) or len(right_eyes) != len(sample_indices):
        raise ValueError("Eye data shape mismatch with sample indices")

    # Add channel dimension and normalize
    batch_data = {
        "left eye": np.expand_dims(left_eyes, axis=-1).astype(np.float32) / 255.0,
        "right eye": np.expand_dims(right_eyes, axis=-1).astype(np.float32) / 255.0,
    }

    return batch_data


def log_batch_progress(
    batch_start: int,
    batch_end: int,
    total_samples: int,
    accepted_count: int,
    confidences: np.ndarray,
) -> None:
    """Log filtering progress with statistics.

    Args:
        batch_start: Starting index of current batch.
        batch_end: Ending index of current batch.
        total_samples: Total number of samples to process.
        accepted_count: Number of samples accepted in this batch.
        confidences: Array of confidence values for this batch.
    """
    progress = (batch_end / total_samples) * 100
    acceptance_rate = accepted_count / len(confidences) if len(confidences) > 0 else 0
    mean_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)

    logger.info(
        f"Filtering progress: {progress:.1f}% - "
        f"Batch {batch_start}-{batch_end-1}: "
        f"{accepted_count}/{len(confidences)} samples accepted "
        f"(rate: {acceptance_rate:.2f}, "
        f"confidence: {mean_confidence:.3f}±{std_confidence:.3f})"
    )


class FilteringProgressTracker:
    """Comprehensive progress tracking for batch filtering."""

    def __init__(self, total_samples: int):
        self.total_samples = total_samples
        self.processed_samples = 0
        self.accepted_samples = 0
        self.rejected_samples = 0
        self.confidences = []
        self.batch_times = []

    def update_batch(
        self,
        batch_start: int,
        batch_end: int,
        accepted_count: int,
        confidences: np.ndarray,
        processing_time: float,
    ):
        """Update progress metrics for a completed batch."""
        self.processed_samples += batch_end - batch_start
        self.accepted_samples += accepted_count
        self.rejected_samples += batch_end - batch_start - accepted_count
        self.confidences.extend(confidences)
        self.batch_times.append(processing_time)

    def get_statistics(self):
        """Get comprehensive filtering statistics."""
        return {
            "progress_percent": (self.processed_samples / self.total_samples) * 100,
            "acceptance_rate": (
                self.accepted_samples / self.processed_samples
                if self.processed_samples > 0
                else 0
            ),
            "mean_confidence": np.mean(self.confidences) if self.confidences else 0,
            "std_confidence": np.std(self.confidences) if self.confidences else 0,
            "total_batches": len(self.batch_times),
            "avg_batch_time": np.mean(self.batch_times) if self.batch_times else 0,
            "estimated_remaining_time": self._estimate_remaining_time(),
        }

    def _estimate_remaining_time(self):
        """Estimate remaining processing time."""
        if not self.batch_times or self.processed_samples == 0:
            return None

        avg_time_per_sample = np.mean(self.batch_times) / DEFAULT_FILTER_BATCH_SIZE
        remaining_samples = self.total_samples - self.processed_samples
        return remaining_samples * avg_time_per_sample


def apply_filter_wrapper_filter(
    dataset: Dict[str, np.ndarray],
    sample_indices: np.ndarray,
    threshold: float,
    batch_size: int = DEFAULT_FILTER_BATCH_SIZE,
) -> np.ndarray:
    """Apply FilterWrapper model to filter samples in batches with progress logging.

    Args:
        dataset: Complete dataset with eye images.
        sample_indices: Array of sample indices to filter.
        threshold: Confidence threshold for acceptance (accept samples with confidence <= threshold).
        min_frames: Minimum frames required for trajectory validation.
        maxT: Maximum time window for trajectory validation.
        batch_size: Number of samples to process in each batch.

    Returns:
        Array of filtered sample indices that passed the quality control.
    """
    if len(sample_indices) == 0:
        logger.warning("No samples to filter")
        return np.array([])

    # Validate threshold (should not be None when called)
    if threshold is None:
        logger.warning("Filter threshold is None, returning all unfiltered samples")
        return sample_indices

    # Validate batch size
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # Initialize FilterWrapper with error handling
    filter_model = load_filter_model()
    if filter_model is None:
        logger.warning(
            "FilterWrapper model not available, returning all unfiltered samples"
        )
        return sample_indices

    # Process in batches with progress tracking
    filtered_indices = []
    total_samples = len(sample_indices)
    progress_tracker = FilteringProgressTracker(total_samples)

    logger.info(
        f"Starting FilterWrapper filtering on {total_samples} samples with threshold {threshold}"
    )

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = sample_indices[batch_start:batch_end]

        processing_start = time.time()

        # Process batch
        batch_data = prepare_filter_batch(dataset, batch_indices)
        predictions = filter_model(batch_data)
        confidences = predictions["predictions"].numpy()

        # Apply threshold: accept low confidence samples (quality control)
        # Lower confidence indicates the model is uncertain about quality, so we keep those
        accepted_mask = confidences.flatten() <= threshold
        accepted_samples = (
            np.array(batch_indices)[accepted_mask]
            if len(batch_indices) > 0
            else np.array([])
        )
        filtered_indices.extend(accepted_samples)

        # Calculate processing time
        processing_time = time.time() - processing_start

        # Update progress tracker
        progress_tracker.update_batch(
            batch_start, batch_end, len(accepted_samples), confidences, processing_time
        )

        # Log progress
        log_batch_progress(
            batch_start, batch_end, total_samples, len(accepted_samples), confidences
        )

    # Log final statistics
    stats = progress_tracker.get_statistics()
    logger.info(f"Filtering statistics: {stats}")

    filtered_array = np.array(filtered_indices)
    retention_rate = (len(filtered_array) / total_samples) * 100

    logger.info(
        f"Filtering complete: {len(filtered_array)}/{total_samples} samples remaining "
        f"({retention_rate:.1f}% retention)"
    )

    return filtered_array


def processFolder(
    folder: str,
    testRatio: float,
    minimumFrames: int,
    dropZeroDeltas: bool,
    maxT: float = 1.0,
    random_seed: Optional[int] = None,
    filter_threshold: float = None,
) -> Tuple[int, int, bool, Any]:
    """Process dataset folder and generate train/test split.

    Args:
        folder: Path to dataset folder.
        testRatio: Fraction of samples for testing.
        minimumFrames: Minimum frames required for trajectory validation.
        dropZeroDeltas: Whether to drop frames with zero time deltas.
        maxT: Maximum time window for trajectory validation (seconds).
        random_seed: Seed for reproducible random split (None for system randomness).
        use_filter: Whether to apply FilterWrapper filtering for quality control (deprecated, use filter_threshold).
        filter_threshold: Confidence threshold for filtering (accept samples with confidence <= threshold, None to disable filtering).

    Returns:
        Tuple of (test_frame_count, train_frame_count, is_skipped, stats_dict).

    Raises:
        AssertionError: If data validation fails.
        ValueError: If invalid parameters are provided.
    """
    # Validate input parameters
    if testRatio <= 0 or testRatio >= 1:
        raise ValueError(f"testRatio must be between 0 and 1, got {testRatio}")
    if minimumFrames < 1:
        raise ValueError(f"minimumFrames must be >= 1, got {minimumFrames}")
    if maxT <= 0:
        raise ValueError(f"maxT must be positive, got {maxT}")
    if filter_threshold is not None and (filter_threshold < 0 or filter_threshold > 1):
        raise ValueError(
            f"filter_threshold must be between 0 and 1, got {filter_threshold}"
        )

    logger.info(f"Processing folder: {folder}")
    stats: Dict[str, List[Any]] = {
        "deltas": [],
        "durations": [],
    }
    folder_path = Path(folder)

    # Load dataset
    all_file = folder_path / "all.npz"
    if all_file.exists():
        dataset = loadNpz(str(all_file))
    else:
        dataset = loadNpz(folder)
        np.savez(str(all_file), **dataset)

    # Remove the npz files, except for all.npz
    removed_count = 0
    for npz_file in folder_path.glob("*.npz"):
        if npz_file.name != "all.npz":
            npz_file.unlink()
            removed_count += 1
    logger.info(f"Removed {removed_count} npz files")

    if dropZeroDeltas:
        dataset = dropZeroTimeDelta(dataset)

    N = len(dataset["time"])
    logger.info(f"Dataset: {N} frames")

    if N < minimumFrames:
        logger.warning("Dataset is too short. Skipping...")
        return 0, 0, True, {}

    # Validate sparsity using refactored module
    try:
        validate_dataset_sparsity(dataset["time"], MAX_DELTA_THRESHOLD)
    except ValueError as e:
        logger.warning(f"Dataset validation failed: {e}")
        return 0, 0, True, {}

    deltas_arr = np.diff(dataset["time"])
    logger.info(
        f"Total time deltas: min={np.min(deltas_arr)}, max={np.max(deltas_arr)}, mean={np.mean(deltas_arr)}"
    )

    # ========== STEP 0-1: Identify valid samples ==========
    # Create filtered dataset for all samples (auto-validates via SampleFilter)
    filtered = create_filtered_dataset(
        dataset, np.arange(len(dataset["time"])), minimumFrames, maxT
    )

    all_valid = filtered.valid_indices()
    if len(all_valid) == 0:
        logger.warning("No valid samples found!")
        return 0, 0, True, {}

    logger.info(f"Total frames: {N}, Valid samples: {len(all_valid)}")

    # ========== STEP 0-2: Apply FilterWrapper filtering ==========
    if filter_threshold is not None:
        # Check if dataset has required eye data for filtering
        if "left eye" not in dataset or "right eye" not in dataset:
            logger.warning(
                "Dataset missing eye data required for filtering, skipping filtering"
            )
        else:
            logger.info("Applying FilterWrapper filtering...")
            original_count = len(all_valid)
            all_valid = apply_filter_wrapper_filter(
                dataset, all_valid, filter_threshold
            )
            remaining_count = len(all_valid)
            filtering_rate = (
                (remaining_count / original_count) * 100 if original_count > 0 else 0
            )

            logger.info(
                f"Filtering complete: {remaining_count}/{original_count} samples remaining "
                f"({filtering_rate:.1f}% retention)"
            )

            if len(all_valid) == 0:
                logger.warning(
                    "No valid samples found after filtering! Dataset will be skipped."
                )
                return 0, 0, True, {}

    # ========== STEP 1: Final validation after filtering ==========
    logger.info(f"Total frames: {N}, Valid samples: {len(all_valid)}")

    # ========== STEP 2: Random split at sample level ==========
    rng = np.random.RandomState(seed=random_seed)
    shuffled = rng.permutation(all_valid)
    split = int(testRatio * len(shuffled))

    # Sort indices to maintain temporal order in subsets (critical for trajectory validation)
    test_samples = np.sort(shuffled[:split])

    logger.info(f"Valid samples: {len(all_valid)} → Test: {len(test_samples)}")

    # Early check: If either split would have zero samples, skip processing entirely
    if len(test_samples) == 0 or (len(all_valid) - len(test_samples)) == 0:
        logger.warning(
            f"Dataset would have zero samples in either train ({len(all_valid) - len(test_samples)}) or test ({len(test_samples)}) split. Skipping entire processing..."
        )
        return 0, 0, True, {}

    # Filter test samples to ensure minimum spacing (no overlaps after frame expansion)
    # So consecutive test samples must be at least minimumFrames apart
    if len(test_samples) > 0:
        filtered_test = [test_samples[0]]
        for s in test_samples[1:]:
            if s - filtered_test[-1] > minimumFrames:
                filtered_test.append(s)
        test_samples = np.array(filtered_test, dtype=np.int64)

    if len(test_samples) == 0:
        logger.warning(
            f"No valid test samples found after filtering! Test ratio: {testRatio}, minimum frames: {minimumFrames}. Dataset will be skipped."
        )
        return 0, 0, True, {}

    train_samples = np.sort(list(set(filtered.used_samples()) - set(test_samples)))
    if len(train_samples) == 0:
        logger.warning(
            "No valid train samples found after filtering! Dataset will be skipped."
        )
        return 0, 0, True, {}

    logger.info(
        f"Valid samples: {len(all_valid)} → Filtered test: {len(test_samples)}, training: {len(train_samples)}"
    )
    # Transform test samples: expand each sample into minimumFrames copies,
    # subtract np.arange(minimumFrames), and collect unique sorted indices
    test_samples = expand_indices(test_samples, minimumFrames)

    # ========== STEP 3: Create clean train/test datasets ==========
    train_filtered = create_filtered_dataset(
        dataset, train_samples, minimumFrames, maxT
    )
    test_filtered = create_filtered_dataset(dataset, test_samples, minimumFrames, maxT)

    # ========== STEP 4: Extract frame unions ==========
    # used_samples returns indices relative to train_data/test_data subsets
    # These are positional indices into the sorted subset arrays
    train_frames_in_subset = np.array(train_filtered.used_samples())
    test_frames_in_subset = np.array(test_filtered.used_samples())

    # Map subset positional indices back to original dataset indices
    train_frames = train_samples[train_frames_in_subset]
    test_frames = test_samples[test_frames_in_subset]

    logger.info(f"Train frames: {len(train_frames)}, Test frames: {len(test_frames)}")

    # ========== STEP 5: Save datasets ==========
    if (0 == len(train_frames)) or (0 == len(test_frames)):
        logger.warning(
            f"No training or testing frames found! Train frames: {len(train_frames)}, Test frames: {len(test_frames)}. Dataset will be skipped."
        )
        return 0, 0, True, None

    def saveSubset(filename: str, idx: np.ndarray, sample_count: int) -> int:
        """Save dataset subset to file with validation."""
        if len(idx) == 0:
            return 0
        logger.info(f"{filename}: {len(idx)} frames from {sample_count} samples")
        subset = {k: v[idx] for k, v in dataset.items()}
        time = subset["time"]
        diff = np.diff(time)
        assert np.all(
            diff >= 0
        ), f"Time not monotonic in {filename}: {np.where(diff < 0)}"
        np.savez(str(folder_path / filename), **subset)
        return len(idx)

    test_count = saveSubset("test.npz", test_frames, len(test_samples))
    train_count = saveSubset("train.npz", train_frames, len(train_samples))

    shapes_str = ", ".join([f"{k}: {v.shape}" for k, v in dataset.items()])
    logger.debug(f"Dataset shapes: {shapes_str}")

    logger.info(f"Processing {folder} done")
    return test_count, train_count, False, stats


def foldersList(x: Union[str, Path]) -> List[str]:
    """Get list of immediate subdirectory names in a folder.

    Lists only direct children directories, not recursive.

    Args:
        x: Path to folder (string or Path object).

    Returns:
        List of subdirectory names in alphabetical order.

    Example:
        >>> dirs = foldersList("data/")
        >>> assert all(isinstance(d, str) for d in dirs)
    """
    x_path = Path(x)
    if not x_path.exists():
        return []
    return sorted([d.name for d in x_path.iterdir() if d.is_dir()])


def traverse_hierarchy(
    folder: Path,
    hierarchy_levels: List[str],
    callback: Callable[[Dict[str, str], Path], None],
    stats: Dict[str, List[Any]],
) -> None:
    """Recursively traverse folder hierarchy and call callback for leaf folders.

    Generalizes nested loop iteration over a multi-level folder structure.

    Args:
        folder: Root folder path to start traversal.
        hierarchy_levels: List of level names (e.g., ["userId", "screenId", "cameraId", "monitorId", "placeId"]).
        callback: Function to call with (level_dict, current_path) at each leaf level.
        stats: Statistics dictionary to collect unique values per level.

    Example:
        >>> def callback(levels, path): print(levels)
        >>> traverse_hierarchy(Path("data/"), ["userId", "screenId"], callback, {})
    """

    def recurse(
        current_path: Path, level_index: int, levels_dict: Dict[str, str]
    ) -> None:
        """Recursively traverse hierarchy levels.

        Args:
            current_path: Current folder path.
            level_index: Current level in hierarchy_levels.
            levels_dict: Dictionary of collected level values so far.

        Side effects:
            Updates stats dict with unique values at each level.
            Calls callback function at leaf level.
        """
        if level_index >= len(hierarchy_levels):
            # Reached leaf level, call callback
            callback(levels_dict, current_path)
            return

        level_name = hierarchy_levels[level_index]
        level_values = foldersList(current_path)

        for value in level_values:
            if value not in stats[level_name]:
                stats[level_name].append(value)

            new_levels_dict = {**levels_dict, level_name: value}
            new_path = current_path / value
            recurse(new_path, level_index + 1, new_levels_dict)

    recurse(folder, 0, {})


def main(args: argparse.Namespace) -> None:
    """Main entry point for preprocessing remote dataset.

    Processes hierarchical folder structure (UserId/ScreenId/CameraId/MonitorId/PlaceId) and generates
    train.npz and test.npz files with train/test split and optional padding.

    Args:
        args: Command-line arguments including folder, test_ratio, minimum_frames, drop_zero_deltas, maxT, random_seed.

    Returns:
        None (writes train.npz, test.npz, and stats.json files).

    Raises:
        AssertionError: If data validation fails.
        FileNotFoundError: If input folder does not exist.
    """
    stats: Dict[str, Any] = {
        "userId": [],
        "screenId": [],
        "cameraId": [],
        "monitorId": [],
        "placeId": [],
        "blacklist": [],
    }
    testFrames = trainFrames = 0
    framesPerChunk: Dict[str, int] = {}
    # subfolders: UserId -> ScreenId -> CameraId -> MonitorId -> PlaceId -> *.npz
    folder = Path(args.folder)
    hierarchy_levels = ["userId", "screenId", "cameraId", "monitorId", "placeId"]
    globalStats: Dict[str, List[Any]] = {
        "deltas": [],
        "durations": [],
    }
    dataset_count: List[int] = [0]  # Use list to allow modification in nested callback

    def process_dataset_folder(levels_dict: Dict[str, str], current_path: Path) -> None:
        """Process a dataset folder at leaf level of hierarchy.

        Args:
            levels_dict: Dictionary with all hierarchy level values.
            current_path: Path to the current folder.

        Side effects:
            Updates nonlocal testFrames, trainFrames, and dataset statistics.
            Writes train.npz and test.npz files to current_path.
        """
        nonlocal testFrames, trainFrames

        userId = levels_dict["userId"]
        screenId = levels_dict["screenId"]
        cameraId = levels_dict["cameraId"]
        monitorId = levels_dict["monitorId"]
        placeId = levels_dict["placeId"]

        testFramesN, trainFramesN, isSkipped, new_stats = processFolder(
            str(current_path),
            args.test_ratio,
            minimumFrames=args.minimum_frames,
            dropZeroDeltas=args.drop_zero_deltas,
            maxT=args.maxT,
            random_seed=args.random_seed,
            filter_threshold=args.filter_threshold,
        )
        if isSkipped:
            logger.warning(
                f"Ignoring dataset: {userId}/{screenId}/{cameraId}/{monitorId}/{placeId} (insufficient valid samples or zero samples in train/test split)"
            )
            # Add ignored folders to blacklist (store actual IDs)
            stats["blacklist"].append([userId, screenId, cameraId, monitorId, placeId])
        else:
            dataset_count[0] += 1
            testFrames += testFramesN
            trainFrames += trainFramesN
            # Store frame counts per chunk using full path for distinction
            dataset_path = DatasetPath(userId, screenId, cameraId, monitorId, placeId)
            sid = dataset_path.full_path
            framesPerChunk[sid] = testFramesN + trainFramesN
            for k, v in new_stats.items():
                globalStats[k].extend(v)

    traverse_hierarchy(folder, hierarchy_levels, process_dataset_folder, stats)
    logger.info(f"Total: {trainFrames} training frames, {testFrames} testing frames")

    # sort each list in stats to preserve the order between runs
    for k, v in stats.items():
        stats[k] = sorted(v)
    # save the stats
    stats_file = folder / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Frame counts per chunk:")
    for k, v in framesPerChunk.items():
        logger.info(f"  {k}: {v} frames")
    ###########################################

    for k, v in globalStats.items():
        if 0 == len(v):
            continue
        v = np.concatenate(v)
        plot_histogram(
            data=v, title=f"Histogram of {k}", filename=str(folder / f"{k}.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument(
        "--folder",
        type=str,
        default=str(ROOT_FOLDER / "Data" / "remote"),
        help="Folder with the npz files",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Ratio of testing samples"
    )
    parser.add_argument(
        "--minimum-frames",
        type=int,
        default=5,
        help="Minimum number of frames in trajectory for sample validation",
    )
    parser.add_argument(
        "--drop-zero-deltas",
        default=True,
        action="store_false",
        help="Drop frames with zero time deltas",
    )
    parser.add_argument(
        "--maxT",
        type=float,
        default=1.0,
        help="Maximum time window for sample trajectory validation (seconds)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible train/test split (None for system randomness)",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=None,
        help="Confidence threshold for FilterWrapper filtering (apply filtering if specified, accept samples with confidence <= threshold, default: disabled)",
    )

    args = parser.parse_args()
    main(args)
