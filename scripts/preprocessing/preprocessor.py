"""Main preprocessor class for orchestrating dataset preprocessing operations.

Coordinates loading, filtering, validation, and trajectory index expansion
for multi-level hierarchical datasets.
"""

from typing import Dict, Tuple
import numpy as np
from Core.logging_config import get_logger
from scripts.Constants import MAX_DELTA_THRESHOLD, DEFAULT_MIN_FRAMES
from scripts.preprocessing.core import (
    remove_frames_with_zero_time_delta,
    expand_indices_for_trajectories,
    validate_dataset_sparsity,
)
from scripts.preprocessing.validation import (
    validate_dataset_structure,
    validate_indices_in_range,
    validate_indices_are_sorted,
)

logger = get_logger(__name__)


class DatasetPreprocessor:
    """Orchestrates preprocessing of hierarchical dataset structure.

    Responsibility chain:
    1. Load datasets from storage
    2. Remove frames with zero/negative time deltas
    3. Validate temporal density (sparsity check)
    4. Expand sample indices to include trajectory frame windows
    5. Provide access to filtered/expanded datasets

    Key invariant: min_trajectory_frames > 0 (enforced in __init__)
    """

    def __init__(
        self,
        min_trajectory_frames: int = DEFAULT_MIN_FRAMES,
        max_delta_threshold: float = MAX_DELTA_THRESHOLD,
    ):
        """Initialize preprocessor with configuration.

        Args:
            min_trajectory_frames: Minimum frames required before each sample
                                  for trajectory context. Must be > 0.
            max_delta_threshold: Maximum allowed time delta between consecutive
                                frames. Used for sparsity validation.

        Raises:
            ValueError: If min_trajectory_frames <= 0.
        """
        if min_trajectory_frames <= 0:
            raise ValueError(
                f"min_trajectory_frames must be positive, got {min_trajectory_frames}"
            )

        self.min_trajectory_frames = min_trajectory_frames
        self.max_delta_threshold = max_delta_threshold

        logger.info(
            f"DatasetPreprocessor initialized: "
            f"min_trajectory_frames={min_trajectory_frames}, "
            f"max_delta_threshold={max_delta_threshold}"
        )

    def process_dataset(
        self,
        dataset: Dict[str, np.ndarray],
        validate_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Process dataset through full filtering pipeline.

        Pipeline:
        1. Validate structure (required keys, array lengths)
        2. Remove frames with zero/negative time deltas
        3. Validate sparsity (temporal density)

        Args:
            dataset: Raw dataset with 'time' and other keys with parallel arrays.
            validate_only: If True, only validate without modifying.

        Returns:
            Filtered dataset (same structure, fewer frames).

        Raises:
            ValueError: If validation fails at any step.
        """
        logger.info(f"Starting dataset processing (validate_only={validate_only})")

        # Step 1: Validate input structure
        validate_dataset_structure(dataset, required_keys=["time"])
        original_count = len(dataset["time"])

        # Step 2: Remove zero-delta frames
        if not validate_only:
            dataset = remove_frames_with_zero_time_delta(dataset)
            filtered_count = len(dataset["time"])
            dropped = original_count - filtered_count
            logger.info(
                f"After zero-delta removal: {filtered_count} frames "
                f"({dropped} removed from {original_count})"
            )
        else:
            logger.info("Validation only: skipping zero-delta removal")

        # Step 3: Validate sparsity
        validate_dataset_sparsity(dataset["time"], self.max_delta_threshold)

        logger.info(f"Dataset processing complete: {len(dataset['time'])} frames")
        return dataset

    def expand_sample_indices(
        self,
        sample_indices: np.ndarray,
        dataset_size: int,
    ) -> np.ndarray:
        """Expand sample indices to include trajectory frame windows.

        For each sample at index i, computes the union of all frames needed:
        [i - min_trajectory_frames + 1, ..., i]

        Args:
            sample_indices: Sorted array of sample start indices.
            dataset_size: Total size of filtered dataset (for validation).

        Returns:
            Sorted unique array of all frame indices needed for trajectories.

        Raises:
            ValueError: If any trajectory expansion results in negative indices.
        """
        logger.info(
            f"Expanding {len(sample_indices)} sample indices "
            f"with min_trajectory_frames={self.min_trajectory_frames}"
        )

        # Validate input indices
        validate_indices_are_sorted(sample_indices, array_name="sample_indices")
        validate_indices_in_range(
            sample_indices,
            min_allowed=self.min_trajectory_frames - 1,
            max_allowed=dataset_size - 1,
            array_name="sample_indices",
        )

        # Expand to frame indices
        frame_indices = expand_indices_for_trajectories(
            sample_indices, self.min_trajectory_frames
        )

        logger.info(
            f"Expansion complete: {len(sample_indices)} samples → "
            f"{len(frame_indices)} frame indices"
        )

        return frame_indices

    def compute_sample_frame_mapping(
        self,
        sample_indices: np.ndarray,
    ) -> Dict[int, Tuple[int, int]]:
        """Compute the frame range [start, end) for each sample's trajectory window.

        For sample at index i, the frame range is:
        [i - min_trajectory_frames + 1, i + 1)

        This is useful for efficiently extracting frame context for each sample.

        Args:
            sample_indices: Sorted array of sample indices.

        Returns:
            Dict mapping sample_index -> (frame_start, frame_end)
            where frame_end is exclusive (range is [frame_start, frame_end))

        Example:
            >>> mapping = compute_sample_frame_mapping(np.array([10, 20, 30]))
            >>> mapping[10]  # Sample at index 10
            (6, 11)  # Frames 6-10 inclusive (using min_trajectory_frames=5)
        """
        mapping = {}
        for sample_idx in sample_indices:
            frame_start = sample_idx - self.min_trajectory_frames + 1
            frame_end = sample_idx + 1  # Exclusive
            mapping[int(sample_idx)] = (frame_start, frame_end)

        logger.info(
            f"Computed frame mapping for {len(sample_indices)} samples "
            f"(min_trajectory_frames={self.min_trajectory_frames})"
        )

        return mapping

    def get_trajectory_context(
        self,
        dataset: Dict[str, np.ndarray],
        sample_index: int,
    ) -> Dict[str, np.ndarray]:
        """Extract trajectory context (frame window) for a single sample.

        Args:
            dataset: Full filtered dataset with 'time' and other array fields.
            sample_index: Index of sample (into dataset arrays).

        Returns:
            Sub-dictionary with same keys, containing frames from
            [sample_index - min_trajectory_frames + 1, sample_index].

        Raises:
            ValueError: If sample_index too close to start of dataset.
        """
        frame_start = sample_index - self.min_trajectory_frames + 1

        if frame_start < 0:
            raise ValueError(
                f"Sample at index {sample_index} requires trajectory starting at "
                f"frame {frame_start}, but dataset starts at 0. "
                f"Need min_trajectory_frames={self.min_trajectory_frames} "
                f"frames before sample."
            )

        frame_end = sample_index + 1

        context = {}
        for key, arr in dataset.items():
            if isinstance(arr, np.ndarray):
                context[key] = arr[frame_start:frame_end]
            else:
                context[key] = arr  # Pass through non-array values

        logger.debug(
            f"Extracted trajectory context for sample {sample_index}: "
            f"frames [{frame_start}, {frame_end})"
        )

        return context

    def validate_train_test_split(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> None:
        """Validate that train/test split is disjoint and complete.

        At SAMPLE level: train_indices ∩ test_indices = ∅
        Together: train_indices ∪ test_indices should be inspectable

        Args:
            train_indices: Sorted array of training sample indices.
            test_indices: Sorted array of testing sample indices.

        Raises:
            ValueError: If sets overlap.
        """
        overlap = np.intersect1d(train_indices, test_indices)

        if len(overlap) > 0:
            raise ValueError(
                f"Train/test split has overlap at sample level: "
                f"{len(overlap)} common sample indices. "
                f"First few: {overlap[:5]}"
            )

        logger.info(
            f"Train/test split validated: "
            f"train={len(train_indices)}, test={len(test_indices)}, "
            f"overlap=0 (disjoint at sample level)"
        )
