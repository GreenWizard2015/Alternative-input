"""Core preprocessing functions for dataset manipulation.

Pure functions for frame filtering, index expansion, and data validation.
These functions are designed to be independently testable and reusable.
"""

from typing import Dict
import numpy as np
from Core.logging_config import get_logger

logger = get_logger(__name__)


def expand_indices_for_trajectories(
    sample_indices: np.ndarray, min_trajectory_frames: int
) -> np.ndarray:
    """Compute frame indices supporting trajectories for given sample start indices.

    For each sample at index i, a valid trajectory requires min_trajectory_frames
    frames before it (i - min_trajectory_frames + 1 through i). This function
    computes the union of all such frame indices.

    OUTPUT INVARIANT: Result is SORTED ASCENDING (enforced by np.unique()).

    Args:
        sample_indices: Sorted array of sample start indices (indices into dataset).
                       Example: [10, 20, 30]
        min_trajectory_frames: Minimum frames required per sample trajectory.
                              Example: 5 requires frames [i-4, ..., i]

    Returns:
        Sorted unique array of all frame indices needed.
        Example: [5,6,7,8,9,10,15,16,17,18,19,20,25,26,27,28,29,30]

    Raises:
        ValueError: If min_trajectory_frames <= 0 or if trajectory expansion
                   results in negative indices (boundary violation).

    Example:
        >>> indices = np.array([10, 20, 30])
        >>> expanded = expand_indices_for_trajectories(indices, 5)
        >>> # Each sample needs 5 frames: [6,7,8,9,10], [16,17,18,19,20], [26,27,28,29,30]
        >>> # Result: [6,7,8,9,10,16,17,18,19,20,26,27,28,29,30]
        >>> assert np.all(expanded[:-1] < expanded[1:])  # Verify sorted
    """
    if min_trajectory_frames <= 0:
        raise ValueError(
            f"min_trajectory_frames must be positive, got {min_trajectory_frames}"
        )

    # Calculate frame range for each sample
    # For sample at index i, need frames: [i - min_trajectory_frames + 1, ..., i]
    min_range = np.arange(min_trajectory_frames)
    expanded = np.repeat(sample_indices, min_trajectory_frames) - np.tile(
        min_range, len(sample_indices)
    )

    # np.unique() automatically sorts and deduplicates
    result = np.unique(expanded)

    # Verify no negative indices (boundary violation)
    if np.any(result < 0):
        min_problematic = np.min(result[result < 0])
        raise ValueError(
            f"expand_indices_for_trajectories() computed negative indices: {min_problematic}. "
            f"This indicates a sample is too close to dataset start. "
            f"Calling code must ensure all samples are >= {min_trajectory_frames} "
            f"from dataset start."
        )

    # Verify sort invariant (np.unique should guarantee this, but double-check)
    assert np.all(
        result[:-1] < result[1:]
    ), "INTERNAL ERROR: expand_indices() result is not sorted"

    return result


def remove_frames_with_zero_time_delta(
    dataset: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Remove frames with zero or negative time deltas.

    Iteratively removes frames where timestamps don't strictly increase,
    recalculating deltas after each removal until stability. Uses STRICT
    comparison (delta > 0.0, no epsilon tolerance).

    Args:
        dataset: Dict with 'time' key containing timestamps and other keys
                with parallel arrays (e.g., 'data', 'landmarks', etc.).

    Returns:
        Filtered dataset with only frames having positive time deltas.
        All arrays filtered to same indices.

    Raises:
        ValueError: If 'time' key not in dataset, if initial times not
                   monotonically increasing, or if dataset is empty.

    Example:
        >>> dataset = {
        ...     'time': np.array([1.0, 1.0, 2.0, 2.5]),
        ...     'data': np.array([10, 11, 12, 13])
        ... }
        >>> result = remove_frames_with_zero_time_delta(dataset)
        >>> # Removes both 1.0 duplicates in first iteration, then stabilizes
        >>> result['time']  # [1.0, 2.0, 2.5] or similar stable state
        >>> assert len(result['data']) == len(result['time'])
    """
    if "time" not in dataset:
        raise ValueError("Dataset must contain 'time' key for filtering")

    initial_times = dataset["time"]
    if len(initial_times) == 0:
        raise ValueError("Dataset 'time' array is empty")

    if len(initial_times) > 1 and not np.all(np.diff(initial_times) >= 0):
        raise ValueError(
            "Time values must be monotonically increasing "
            f"(non-decreasing). Found violations: "
            f"min_delta={np.min(np.diff(initial_times))}"
        )

    # Create mutable copy to avoid modifying input
    result = {
        k: v.copy() if isinstance(v, np.ndarray) else v for k, v in dataset.items()
    }

    iteration = 0
    while True:
        iteration += 1
        original_count = len(result["time"])

        # Calculate time deltas between consecutive frames
        deltas = np.diff(result["time"])

        # Find frames to REMOVE: where delta[i] <= 0 means frame[i+1] is bad
        # (Because delta[i] = time[i+1] - time[i])
        invalid_frame_indices = np.where(deltas <= 0.0)[0] + 1  # +1 to get frame index
        dropped_count = len(invalid_frame_indices)

        # Log statistics
        if len(deltas) > 0:
            logger.info(
                f"Iteration {iteration}: Time deltas - "
                f"min={np.min(deltas):.6f}, "
                f"max={np.max(deltas):.6f}, "
                f"mean={np.mean(deltas):.6f}"
            )
        logger.info(
            f"Iteration {iteration}: Dropping {dropped_count} frames "
            f"with zero/negative deltas, keeping {original_count - dropped_count} frames"
        )

        # If no frames to drop, we're done
        if dropped_count == 0:
            logger.info(
                f"Converged after {iteration} iteration(s). "
                f"Final dataset: {len(result['time'])} frames"
            )
            return result

        # Remove invalid frames using setdiff1d to get remaining indices
        all_indices = np.arange(original_count)
        valid_indices = np.setdiff1d(all_indices, invalid_frame_indices)

        # Filter all arrays to valid indices
        result = {
            k: v[valid_indices] if isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }

        # Continue iteration loop


def validate_dataset_sparsity(
    time_array: np.ndarray, max_delta_threshold: float
) -> None:
    """Validate that dataset is not too sparse (frames have sufficient temporal density).

    Checks that the minimum time delta between consecutive frames does not
    exceed threshold. If dataset is too sparse, raises ValueError.

    Args:
        time_array: Monotonically increasing array of timestamps (seconds).
        max_delta_threshold: Maximum allowed minimum delta between frames (seconds).
                            Example: 0.3 means frames must be within 300ms

    Returns:
        None (raises exception if validation fails)

    Raises:
        ValueError: If dataset has fewer than 2 frames or if minimum
                   frame delta exceeds threshold.

    Example:
        >>> times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> validate_dataset_sparsity(times, max_delta_threshold=0.3)
        >>> # Passes - min delta is 0.1 < 0.3

        >>> sparse_times = np.array([0.0, 1.0, 2.0])
        >>> validate_dataset_sparsity(sparse_times, max_delta_threshold=0.3)
        >>> # Raises ValueError - min delta is 1.0 > 0.3
    """
    if len(time_array) < 2:
        raise ValueError(
            f"Dataset must have at least 2 frames for sparsity check, "
            f"got {len(time_array)}"
        )

    deltas = np.diff(time_array)
    min_delta = np.min(deltas)

    if min_delta > max_delta_threshold:
        raise ValueError(
            f"Dataset is too sparse: minimum frame delta {min_delta:.3f}s "
            f"exceeds threshold {max_delta_threshold}s. "
            f"Frames must have temporal density (max gap {max_delta_threshold}s)"
        )

    logger.info(
        f"Dataset sparsity check passed: "
        f"min_delta={min_delta:.3f}s (threshold={max_delta_threshold}s), "
        f"total_duration={time_array[-1] - time_array[0]:.1f}s, "
        f"frame_count={len(time_array)}"
    )
