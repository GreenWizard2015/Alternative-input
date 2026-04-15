"""Dataset loading utilities for reading .npz files and extracting sessions.

Provides functions for loading datasets from folders or single files,
extracting temporal sessions based on time gaps, and counting samples.
"""

from typing import Dict, Tuple, List, Iterator, Optional, Any, NamedTuple
import numpy as np
from collections import defaultdict
import os
import glob
import json
import itertools

from Core.utils.DatasetPath import DatasetPath
import Core.Constants as Constants


class DatasetInfo(NamedTuple):
    """Information about a dataset folder with path and indices.

    Attributes:
        path: DatasetPath object for accessing folder paths
        user_idx: Integer index of user in sorted stats["userId"] list
        screen_idx: Integer index of screen in sorted stats["screenId"] list
        camera_idx: Integer index of camera in sorted stats["cameraId"] list
        monitor_idx: Integer index of monitor in sorted stats["monitorId"] list
        place_idx: Integer index of place in sorted stats["placeId"] list
    """

    path: DatasetPath
    user_idx: int
    screen_idx: int
    camera_idx: int
    monitor_idx: int
    place_idx: int


def data_from_folder(folder: str) -> Iterator[Dict[str, np.ndarray]]:
    """Iterate over .npz files in folder.

    Yields data from each .npz file in the specified folder, loading them
    sequentially to avoid excessive memory usage.

    Args:
        folder: Path to folder containing .npz files

    Yields:
        Dictionary for each .npz file with numpy arrays
    """
    for fn in glob.iglob(os.path.join(folder, "*.npz")):
        with np.load(fn) as data:
            yield data


def dataset_from(folder: str) -> Optional[Dict[str, np.ndarray]]:
    """Load dataset from folder or single .npz file.

    Loads data from either a single .npz file or a folder of .npz files,
    concatenating them into a single dataset. Sorts by time if needed.

    Args:
        folder: Path to folder containing .npz files or path to single .npz file

    Returns:
        Dictionary with concatenated numpy arrays, or None if folder is empty
    """
    result = None
    if os.path.isfile(folder):
        with np.load(folder) as data:
            result = {k: v for k, v in data.items()}
    else:
        dataset = defaultdict(list)
        for data in data_from_folder(folder):
            for k, v in data.items():
                dataset[k].append(v)

        result = {k: np.concatenate(v, axis=0) for k, v in dataset.items()}

    if not result:
        return None

    time_sort_indices = np.argsort(result["time"])
    if not all(
        sorted_index == position_index
        for position_index, sorted_index in enumerate(time_sort_indices)
    ):
        result = {k: v[time_sort_indices] for k, v in result.items()}

    return result


def extract_sessions(
    dataset: Dict[str, np.ndarray], max_delta: float
) -> List[Tuple[int, int]]:
    """Extract sessions from dataset based on time gaps.

    Segments dataset into sessions where consecutive samples are within max_delta
    seconds. Sessions with less than 2 samples are filtered out.

    Args:
        dataset: Dataset dictionary containing 'time' key with timestamps
        max_delta: Maximum time gap (seconds) within a session

    Returns:
        List of (start_idx, end_idx+1) tuples defining session boundaries

    Raises:
        AssertionError: If sessions overlap (internal consistency check)
    """
    result = []
    last_timestamp = 0
    session_start_idx = 0
    num_samples = len(dataset["time"])
    for current_idx, current_timestamp in enumerate(dataset["time"]):
        if max_delta < (current_timestamp - last_timestamp):
            if 1 < (current_idx - session_start_idx):
                result.append((session_start_idx, current_idx))
            session_start_idx = current_idx
        last_timestamp = current_timestamp
    if session_start_idx < num_samples:
        result.append((session_start_idx, num_samples))

    result = [x for x in result if 1 < (x[1] - x[0])]

    for i in range(1, len(result)):
        assert (
            result[i - 1][1] <= result[i][0]
        ), f"Sessions overlap: {result[i-1]} and {result[i]}"
    return result


def count_samples_in(folder: str) -> int:
    """Count total samples in all .npz files in folder.

    Iterates through all .npz files in the folder and sums up the number of
    samples (determined by 'time' array length).

    Args:
        folder: Path to folder containing .npz files

    Returns:
        Total number of samples across all files
    """
    result = 0
    for fn in glob.iglob(os.path.join(folder, "*.npz")):
        with np.load(fn) as data:
            result += len(data["time"])
    return result


def ensure_sorted_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all ID lists in stats are sorted for deterministic indices.

    Args:
        stats: Statistics dictionary with ID lists

    Returns:
        New dict with all ID lists sorted alphabetically

    Raises:
        ValueError: If required keys missing
    """
    # Validate structure
    missing = set(Constants.HIERARCHY_LEVELS) - set(stats.keys())
    if missing:
        raise ValueError(f"stats missing required keys: {missing}")

    # Sort all ID lists
    sorted_stats = {**stats}
    for key in Constants.HIERARCHY_LEVELS:
        if isinstance(sorted_stats[key], list):
            sorted_stats[key] = sorted(sorted_stats[key])

    return sorted_stats


def read_json(filepath: str, sort_stats: bool = True) -> Dict[str, Any]:
    """Load JSON data from a file, optionally sorting stats lists.

    Args:
        filepath: Path to the JSON file to load
        sort_stats: If True and file contains stats, sort ID lists (default: True)

    Returns:
        Dictionary containing the parsed JSON data with sorted ID lists if applicable

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Auto-sort if this looks like a stats file (has HIERARCHY_LEVELS keys)
    if sort_stats and set(Constants.HIERARCHY_LEVELS).issubset(set(data.keys())):
        data = ensure_sorted_stats(data)

    return data


def dataset_from_stats(json_path: str) -> Iterator[DatasetInfo]:
    """Generate DatasetInfo objects for valid dataset folders from statistics JSON file.

    Reads statistics from JSON file, iterates through all valid combinations of users,
    screens, cameras, monitors, and places, yielding DatasetInfo objects with paths
    and indices for dataset folders that exist and haven't been blacklisted.

    Note:
        ID lists are sorted before enumeration to ensure deterministic indices.
        Without sorting, enumeration indices depend on stats.json order, causing
        the same dataset to get different indices across runs if stats order changes.

    Args:
        json_path: Path to statistics JSON file containing keys:
            - 'userId': List of user IDs
            - 'screenId': List of screen IDs
            - 'cameraId': List of camera IDs
            - 'monitorId': List of monitor IDs
            - 'placeId': List of place IDs
            - 'blacklist': Optional list of [user_id, screen_id, camera_id, monitor_id, place_id] (actual IDs) to skip

    Yields:
        DatasetInfo object for each valid dataset folder containing path and indices
    """
    stats = read_json(json_path)
    folder = os.path.dirname(json_path)

    # Sort ID lists before enumeration to ensure deterministic indices.
    # This prevents index shifts when stats.json order changes.
    sorted_ids = {key: sorted(stats[key]) for key in Constants.HIERARCHY_LEVELS}

    blacklist_set = set(map(tuple, stats.get("blacklist", [])))

    # Create enumerated lists for each hierarchy level: [(0, id_0), (1, id_1), ...]
    enumerated_lists = [
        enumerate(sorted_ids[key]) for key in Constants.HIERARCHY_LEVELS
    ]

    # Generate all combinations: ((user_idx, user_id), (screen_idx, screen_id), ...)
    for combo in itertools.product(*enumerated_lists):
        indices, ids = zip(*combo)

        if ids in blacklist_set:
            continue

        user_idx, screen_idx, camera_idx, monitor_idx, place_idx = indices
        user_id, screen_id, camera_id, monitor_id, place_id = ids

        dataset_path = DatasetPath(
            user_id, screen_id, camera_id, monitor_id, place_id, base_path=folder
        )
        if not os.path.exists(dataset_path.full_path):
            continue

        yield DatasetInfo(
            path=dataset_path,
            user_idx=user_idx,
            screen_idx=screen_idx,
            camera_idx=camera_idx,
            monitor_idx=monitor_idx,
            place_idx=place_idx,
        )
