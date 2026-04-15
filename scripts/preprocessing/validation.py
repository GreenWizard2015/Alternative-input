"""Validation helpers for preprocessing operations.

Provides functions to validate dataset structure, content consistency,
and index validity.
"""

from typing import Dict, List, Optional
import numpy as np
from Core.logging_config import get_logger

logger = get_logger(__name__)


def validate_dataset_structure(
    dataset: Dict[str, np.ndarray], required_keys: Optional[List[str]] = None
) -> None:
    """Validate that dataset has required keys and consistent array lengths.

    Args:
        dataset: Dictionary with array-like values.
        required_keys: List of required keys. If None, only checks for 'time'.

    Raises:
        ValueError: If required keys missing or array lengths inconsistent.
    """
    if required_keys is None:
        required_keys = ["time"]

    # Check all required keys present
    missing_keys = set(required_keys) - set(dataset.keys())
    if missing_keys:
        raise ValueError(
            f"Dataset missing required keys: {missing_keys}. "
            f"Available keys: {set(dataset.keys())}"
        )

    # Get length of first array
    arrays = {k: v for k, v in dataset.items() if isinstance(v, np.ndarray)}
    if not arrays:
        raise ValueError("Dataset contains no numpy arrays")

    first_key = list(arrays.keys())[0]
    expected_length = len(arrays[first_key])

    # Check all arrays have same length
    for key, arr in arrays.items():
        if len(arr) != expected_length:
            raise ValueError(
                f"Array length mismatch: '{first_key}' has {expected_length} elements, "
                f"but '{key}' has {len(arr)} elements"
            )

    logger.info(
        f"Dataset structure validated: {len(arrays)} arrays, "
        f"{expected_length} frames, keys={list(arrays.keys())}"
    )


def validate_indices_in_range(
    indices: np.ndarray,
    min_allowed: int = 0,
    max_allowed: Optional[int] = None,
    array_name: str = "indices",
) -> None:
    """Validate that indices are within allowed range.

    Args:
        indices: Array of indices to validate.
        min_allowed: Minimum allowed index (inclusive). Default 0.
        max_allowed: Maximum allowed index (inclusive). If None, no upper limit.
        array_name: Name of array for error messages.

    Raises:
        ValueError: If any index out of range.
    """
    if len(indices) == 0:
        logger.info(f"{array_name} is empty - no indices to validate")
        return

    min_idx = np.min(indices)
    max_idx = np.max(indices)

    if min_idx < min_allowed:
        raise ValueError(
            f"{array_name}: minimum index {min_idx} < allowed minimum {min_allowed}"
        )

    if max_allowed is not None and max_idx > max_allowed:
        raise ValueError(
            f"{array_name}: maximum index {max_idx} > allowed maximum {max_allowed}"
        )

    logger.info(
        f"{array_name} range validated: [{min_idx}, {max_idx}] "
        f"within [{min_allowed}, {max_allowed}]"
    )


def validate_indices_are_sorted(
    indices: np.ndarray, array_name: str = "indices", allow_duplicates: bool = False
) -> None:
    """Validate that indices are sorted in ascending order.

    Args:
        indices: Array of indices to validate.
        array_name: Name of array for error messages.
        allow_duplicates: If False, require strictly ascending (no duplicates).

    Raises:
        ValueError: If indices not properly sorted.
    """
    if len(indices) <= 1:
        logger.info(f"{array_name} has <= 1 element, trivially sorted")
        return

    if allow_duplicates:
        is_sorted = np.all(indices[:-1] <= indices[1:])
        comparison = "<="
    else:
        is_sorted = np.all(indices[:-1] < indices[1:])
        comparison = "<"

    if not is_sorted:
        diffs = np.diff(indices)
        bad_indices = np.where(diffs <= 0 if allow_duplicates else diffs < 0)[0]
        first_bad = bad_indices[0] if len(bad_indices) > 0 else -1
        raise ValueError(
            f"{array_name} not properly sorted: "
            f"at position {first_bad}, {indices[first_bad]} "
            f"is not {comparison} {indices[first_bad + 1]}"
        )

    logger.info(f"{array_name} is properly sorted (strictly ascending)")


def validate_indices_unique(indices: np.ndarray, array_name: str = "indices") -> None:
    """Validate that all indices are unique (no duplicates).

    Args:
        indices: Array of indices to check.
        array_name: Name of array for error messages.

    Raises:
        ValueError: If duplicate indices found.
    """
    unique_count = len(np.unique(indices))
    total_count = len(indices)

    if unique_count != total_count:
        duplicates = total_count - unique_count
        raise ValueError(
            f"{array_name}: found {duplicates} duplicate values "
            f"({unique_count} unique out of {total_count} total)"
        )

    logger.info(f"{array_name} has no duplicates ({total_count} unique values)")


def validate_no_index_overlap(
    indices1: np.ndarray,
    indices2: np.ndarray,
    name1: str = "indices1",
    name2: str = "indices2",
) -> None:
    """Validate that two index arrays have no overlap (disjoint sets).

    Args:
        indices1: First array of indices.
        indices2: Second array of indices.
        name1: Name of first array for error messages.
        name2: Name of second array for error messages.

    Raises:
        ValueError: If overlap found.
    """
    overlap = np.intersect1d(indices1, indices2)

    if len(overlap) > 0:
        raise ValueError(
            f"{name1} and {name2} overlap: {len(overlap)} common indices "
            f"(first few: {overlap[:5]})"
        )

    logger.info(
        f"{name1} and {name2} are disjoint: "
        f"|{name1}|={len(indices1)}, |{name2}|={len(indices2)}, overlap=0"
    )


def validate_index_subset(
    subset: np.ndarray,
    superset: np.ndarray,
    subset_name: str = "subset",
    superset_name: str = "superset",
) -> None:
    """Validate that subset is indeed a subset of superset.

    Args:
        subset: Array that should be subset.
        superset: Array that should be superset.
        subset_name: Name of subset for error messages.
        superset_name: Name of superset for error messages.

    Raises:
        ValueError: If subset is not contained in superset.
    """
    not_in_superset = np.setdiff1d(subset, superset)

    if len(not_in_superset) > 0:
        raise ValueError(
            f"{subset_name} contains {len(not_in_superset)} elements not in {superset_name}: "
            f"first few: {not_in_superset[:5]}"
        )

    logger.info(
        f"{subset_name} is valid subset of {superset_name}: "
        f"|subset|={len(subset)}, |superset|={len(superset)}"
    )


def validate_partition_complete(
    partition1: np.ndarray,
    partition2: np.ndarray,
    total_size: int,
    name1: str = "partition1",
    name2: str = "partition2",
) -> None:
    """Validate that two partitions are disjoint and together cover total_size elements.

    Args:
        partition1: First partition array.
        partition2: Second partition array.
        total_size: Total number of elements they should cover.
        name1: Name of first partition for error messages.
        name2: Name of second partition for error messages.

    Raises:
        ValueError: If partitions overlap or don't cover exactly total_size.
    """
    # Check disjoint
    overlap = np.intersect1d(partition1, partition2)
    if len(overlap) > 0:
        raise ValueError(
            f"Partitions {name1} and {name2} overlap: {len(overlap)} common elements"
        )

    # Check coverage
    combined_size = len(partition1) + len(partition2)
    if combined_size != total_size:
        raise ValueError(
            f"Partitions {name1} and {name2} don't cover all elements: "
            f"|{name1}|={len(partition1)} + |{name2}|={len(partition2)} = {combined_size}, "
            f"but should equal {total_size}"
        )

    logger.info(
        f"Partition validation: {name1} ({len(partition1)}) + "
        f"{name2} ({len(partition2)}) = {total_size} (complete and disjoint)"
    )


def validate_array_dtype_compatible(
    arr: np.ndarray, expected_dtype_family: str, array_name: str = "array"
) -> None:
    """Validate that array dtype is compatible with expected family.

    Args:
        arr: Array to check.
        expected_dtype_family: One of 'numeric', 'integer', 'float', 'bool'.
        array_name: Name of array for error messages.

    Raises:
        ValueError: If dtype incompatible.
    """
    dtype = arr.dtype

    if expected_dtype_family == "numeric":
        is_compatible = np.issubdtype(dtype, np.integer) or np.issubdtype(
            dtype, np.floating
        )
    elif expected_dtype_family == "integer":
        is_compatible = np.issubdtype(dtype, np.integer)
    elif expected_dtype_family == "float":
        is_compatible = np.issubdtype(dtype, np.floating)
    elif expected_dtype_family == "bool":
        is_compatible = np.issubdtype(dtype, np.bool_)
    else:
        raise ValueError(f"Unknown dtype family: {expected_dtype_family}")

    if not is_compatible:
        raise ValueError(
            f"{array_name} has dtype {dtype}, expected {expected_dtype_family}"
        )

    logger.info(
        f"{array_name} dtype {dtype} is compatible with {expected_dtype_family}"
    )


def validate_monotonic_increasing(
    arr: np.ndarray, array_name: str = "array", strict: bool = True
) -> None:
    """Validate that array is monotonically increasing.

    Args:
        arr: Array to validate.
        array_name: Name of array for error messages.
        strict: If True, require strictly increasing (no duplicates).
               If False, allow duplicates (non-decreasing).

    Raises:
        ValueError: If not monotonic increasing.
    """
    if len(arr) <= 1:
        logger.info(f"{array_name} has <= 1 element, trivially monotonic")
        return

    diffs = np.diff(arr)
    comparison = "strictly increasing" if strict else "non-decreasing"

    if strict:
        is_valid = np.all(diffs > 0)
    else:
        is_valid = np.all(diffs >= 0)

    if not is_valid:
        min_diff = np.min(diffs)
        bad_idx = np.where(diffs <= 0 if strict else diffs < 0)[0][0]
        raise ValueError(
            f"{array_name} is not {comparison}: "
            f"at position {bad_idx}, diff={diffs[bad_idx]} "
            f"(min_diff={min_diff})"
        )

    logger.info(
        f"{array_name} is {comparison}: "
        f"min_delta={np.min(diffs):.6f}, max_delta={np.max(diffs):.6f}"
    )


def validate_no_negative_values(arr: np.ndarray, array_name: str = "array") -> None:
    """Validate that array contains no negative values.

    Args:
        arr: Array to check.
        array_name: Name of array for error messages.

    Raises:
        ValueError: If negative values found.
    """
    if len(arr) == 0:
        logger.info(f"{array_name} is empty")
        return

    if np.any(arr < 0):
        min_val = np.min(arr)
        neg_count = np.sum(arr < 0)
        raise ValueError(
            f"{array_name} contains {neg_count} negative values " f"(min={min_val})"
        )

    logger.info(f"{array_name} contains no negative values")


def validate_array_finite(arr: np.ndarray, array_name: str = "array") -> None:
    """Validate that array contains only finite values (no NaN, Inf).

    Args:
        arr: Array to check.
        array_name: Name of array for error messages.

    Raises:
        ValueError: If non-finite values found.
    """
    if arr.dtype.kind not in ["f", "c"]:  # float or complex
        logger.info(f"{array_name} is not floating-point, skipping finite check")
        return

    if len(arr) == 0:
        logger.info(f"{array_name} is empty")
        return

    inf_count = np.sum(np.isinf(arr))
    nan_count = np.sum(np.isnan(arr))

    if inf_count > 0 or nan_count > 0:
        raise ValueError(
            f"{array_name} contains non-finite values: "
            f"{nan_count} NaN, {inf_count} Inf"
        )

    logger.info(f"{array_name} contains only finite values")
