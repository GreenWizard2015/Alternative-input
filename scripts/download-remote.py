#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Downloads the data from the remote server and saves it in the local folder
In structure:
Data
  remote
    UserId
      ScreenId
        CameraId
          MonitorId
            PlaceId
              *.npz
"""
from typing import Dict, Optional, Tuple, List, Callable, Any
from pathlib import Path
from collections import defaultdict
from Core.Utils import FACE_MESH_POINTS, DatasetPath
import requests
import numpy as np
import shutil
import gzip
import io as IO
import argparse
import os
from Core.logging_config import get_logger

logger = get_logger(__name__)

folder = Path(__file__).parent.parent / "Data"

# Validation constants for deserialized data
POINTS_VALID_MIN = -0.5
POINTS_VALID_MAX = 1.5
GOAL_VALID_MIN = -2.0
GOAL_VALID_MAX = 2.0
EYE_SIZE = 48


def deserialize(buffer: bytes) -> Dict[str, np.ndarray]:
    """Deserialize binary data into samples dictionary.

    Reads eye images, face mesh points, goals, and metadata from compressed binary format.
    Validates points are within valid range [-0.5, 1.5] and goals within [-2, 2].

    Args:
        buffer: Binary buffer containing serialized samples.

    Returns:
        Dictionary with keys: userId, screenId, cameraId, monitorId, placeId, time, left eye, right eye, points, goal.

    Raises:
        ValueError: If version is not 4 (only version 4 supported).
        AssertionError: If point coordinates are invalid.

    Example:
        >>> data = deserialize(buffer)
        >>> assert data["points"].shape == (N, 468, 2)
    """
    offset = 0
    samples = []
    # read header (uint8)
    version = np.frombuffer(buffer, dtype=np.uint8, count=1, offset=offset)[0]
    if version != 4:  # only version 4 is supported
        raise ValueError(f"Invalid version {version}, only version 4 is supported")
    offset += 1

    userId = buffer[offset : offset + 36].decode("utf-8")
    offset += 36

    placeId = buffer[offset : offset + 36].decode("utf-8")
    offset += 36

    screenId = buffer[offset : offset + 36].decode("utf-8")
    offset += 36

    cameraId = buffer[offset : offset + 36].decode("utf-8")
    offset += 36

    monitorId = buffer[offset : offset + 36].decode("utf-8")
    offset += 36

    eye_size = EYE_SIZE
    # read samples
    while offset < len(buffer):
        sample: Dict[str, Any] = {
            "userId": userId,
            "placeId": placeId,
            "screenId": screenId,
            "cameraId": cameraId,
            "monitorId": monitorId,
        }

        # Read time (uint64)
        time_data = np.frombuffer(buffer, dtype=">Q", count=1, offset=offset)
        sample["time"] = time_data[0]
        offset += 8

        # Read leftEye (uint8)
        eye_count = eye_size * eye_size
        left_eye_data: np.ndarray = np.frombuffer(
            buffer, dtype=np.uint8, count=eye_count, offset=offset
        ).reshape(eye_size, eye_size)
        sample["leftEye"] = left_eye_data
        offset += eye_count

        # Read rightEye (uint8)
        right_eye_data: np.ndarray = np.frombuffer(
            buffer, dtype=np.uint8, count=eye_count, offset=offset
        ).reshape(eye_size, eye_size)
        sample["rightEye"] = right_eye_data
        offset += eye_count

        # Read points (float32)
        points_data: np.ndarray = np.frombuffer(
            buffer, dtype=">f4", count=2 * FACE_MESH_POINTS, offset=offset
        ).reshape(FACE_MESH_POINTS, 2)
        sample["points"] = points_data
        # Validate points are within expected range [POINTS_VALID_MIN, POINTS_VALID_MAX]
        points = sample["points"]
        is_valid = (POINTS_VALID_MIN <= points) & (points <= POINTS_VALID_MAX)
        is_valid = is_valid.all(axis=-1)
        invalid_points = points[~is_valid]
        assert np.all(
            is_valid
        ), f"Invalid points outside range [{POINTS_VALID_MIN}, {POINTS_VALID_MAX}]: found {len(invalid_points)} invalid points, examples: {invalid_points[:5]}"
        offset += 4 * 2 * FACE_MESH_POINTS

        # Read goal (float32)
        goal: np.ndarray = np.frombuffer(buffer, dtype=">f4", count=2, offset=offset)
        sample["goal"] = goal
        offset += 4 * 2

        if (GOAL_VALID_MIN < goal).all() and (goal < GOAL_VALID_MAX).all():
            samples.append(sample)
        else:
            logger.warning(f"Invalid goal coordinates: {goal}")

    # Transpose them to make columnwise arrays
    res = {}
    for k in samples[0].keys():
        res[k] = np.array([sample[k] for sample in samples])
        assert res[k].shape[0] == len(
            samples
        ), f"Invalid shape for {k}: expected {len(samples)} samples, got {res[k].shape[0]}"
    # convert the time to float64
    res["time"] = res["time"].astype(np.float64) / 1000.0
    # rename "leftEye" and "rightEye" to "left eye" and "right eye"
    res["left eye"] = res.pop("leftEye")
    res["right eye"] = res.pop("rightEye")
    if 1 == version:  # upscale to 48x48
        import cv2

        res["left eye"] = np.stack(
            [cv2.resize(img[..., None], (48, 48)) for img in res["left eye"]]
        )
        res["right eye"] = np.stack(
            [cv2.resize(img[..., None], (48, 48)) for img in res["right eye"]]
        )

    assert res["left eye"].shape[1:] == (
        48,
        48,
    ), f"Invalid shape for left eye. Expected (N, 48, 48), got {res['left eye'].shape}"
    assert res["right eye"].shape[1:] == (
        48,
        48,
    ), f"Invalid shape for right eye. Expected (N, 48, 48), got {res['right eye'].shape}"

    # Version 4 stores 4 IDs directly, remove them from samples before returning
    # The IDs are already in the sample dict from deserialization
    return res


def find_free_name(folder: str, base_name: str, extension: str = ".npz") -> str:
    """Find available filename by incrementing counter if needed.

    Searches for a free filename in the given folder by appending an incrementing
    counter until a filename that doesn't exist is found.

    Args:
        folder: Directory path to search in.
        base_name: Base filename without extension.
        extension: File extension (default: ".npz").

    Returns:
        Full path to an available filename.

    Example:
        >>> path = find_free_name("output", "data", ".npz")
        >>> assert not Path(path).exists()
    """
    folder_path = Path(folder)
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_name}{extension}"
        else:
            file_name = f"{base_name}_{counter}{extension}"

        file_path = folder_path / file_name
        if not file_path.exists():
            return str(file_path)
        counter += 1


# for (user, screen, camera, place) store (start/end time) to assert in saveChunk there is no overlaps
chunks_ranges: Dict[str, List[Tuple[float, float]]] = defaultdict(list)


def check_time_overlap(chunk_key: str, chunk_start: float, chunk_end: float) -> None:
    """Check for time range overlaps with previously saved chunks.

    Two intervals overlap if NOT (a_end < b_start OR a_start > b_end).
    Equivalently, they overlap if one starts before the other ends AND vice versa.

    Args:
        chunk_key: Unique identifier for (user, screen, camera, place) combination.
        chunk_start: Start time of new chunk.
        chunk_end: End time of new chunk.

    Raises:
        AssertionError: If time range overlaps with existing chunks.

    Example:
        >>> check_time_overlap("user1_screen1", 0.0, 1.0)
        >>> check_time_overlap("user1_screen1", 2.0, 3.0)  # OK, no overlap
    """
    for prev_start, prev_end in chunks_ranges[chunk_key]:
        # Ranges overlap if: NOT (chunk_end < prev_start OR chunk_start > prev_end)
        # Which simplifies to: prev_start < chunk_end AND chunk_start < prev_end
        if prev_start < chunk_end and chunk_start < prev_end:
            raise AssertionError(
                f"Time overlap detected for chunk key '{chunk_key}': "
                f"new chunk time range=[{chunk_start:.1f}, {chunk_end:.1f}], "
                f"existing chunk time range=[{prev_start:.1f}, {prev_end:.1f}]. "
                f"This indicates duplicate or out-of-order data."
            )


def record_chunk_time_range(
    chunk_key: str, chunk_start: float, chunk_end: float
) -> None:
    """Record time range for a chunk to enable future overlap detection.

    Args:
        chunk_key: Unique identifier for (user, place, screen) combination.
        chunk_start: Start time of chunk.
        chunk_end: End time of chunk.

    Example:
        >>> record_chunk_time_range("user1_screen1", 0.0, 1.0)
        >>> record_chunk_time_range("user1_screen1", 2.0, 3.0)
    """
    chunks_ranges[chunk_key].append((chunk_start, chunk_end))
    # Keep ranges sorted by start time for efficient range queries if needed
    chunks_ranges[chunk_key].sort(key=lambda x: x[0])


def saveChunk(samples: Dict[str, np.ndarray], folder: str) -> None:
    """Save samples to compressed NPZ file organized by userId/screenId/cameraId/monitorId/placeId.

    Validates that all samples belong to the same user/screen/camera/monitor/place, timestamps
    are monotonically increasing, and that new chunk time ranges do not overlap
    with previously saved chunks for the same user/screen/camera/monitor/place combination.
    Creates directory structure and saves samples as compressed NPZ file.

    Args:
        samples: Dictionary of sample arrays with userId, screenId, cameraId, monitorId, placeId, time.
        folder: Base folder path where to create userId/screenId/cameraId/monitorId/placeId hierarchy.

    Raises:
        AssertionError: If time not monotonically increasing, multiple IDs in chunk,
                       or time range overlaps with existing chunks.

    Example:
        >>> samples = {"userId": ["u1"], "screenId": ["s1"], ...}
        >>> saveChunk(samples, "data/remote")
    """
    # Validate time is increasing monotonically (within chunk)
    time = samples["time"]
    time_diffs = np.diff(time)
    min_diff = np.min(time_diffs)
    assert np.all(
        time[1:] >= time[:-1]
    ), f"Time not monotonically increasing: {len(time)} samples, min_diff={min_diff}, first_diff={time_diffs[0]}, last_diff={time_diffs[-1]}"

    # Validate sample consistency: all samples must belong to same user/screen/camera/monitor/place
    userId = np.unique(samples["userId"])
    assert 1 == len(
        userId
    ), f"Expected 1 userId in chunk, got {len(userId)} unique values: {userId}"
    screenId = np.unique(samples["screenId"])
    assert 1 == len(
        screenId
    ), f"Expected 1 screenId in chunk, got {len(screenId)} unique values: {screenId}"
    cameraId = np.unique(samples["cameraId"])
    assert 1 == len(
        cameraId
    ), f"Expected 1 cameraId in chunk, got {len(cameraId)} unique values: {cameraId}"
    monitorId = np.unique(samples["monitorId"])
    assert 1 == len(
        monitorId
    ), f"Expected 1 monitorId in chunk, got {len(monitorId)} unique values: {monitorId}"
    placeId = np.unique(samples["placeId"])
    assert 1 == len(
        placeId
    ), f"Expected 1 placeId in chunk, got {len(placeId)} unique values: {placeId}"

    # Validate no time range overlap with previously saved chunks
    chunk_key = f"{userId[0]}_{screenId[0]}_{cameraId[0]}_{monitorId[0]}_{placeId[0]}"
    chunk_start = time[0]
    chunk_end = time[-1]

    # Check for overlaps with existing chunks
    check_time_overlap(chunk_key, chunk_start, chunk_end)
    record_chunk_time_range(chunk_key, chunk_start, chunk_end)

    # Use DatasetPath for consistent path handling
    dataset_path = DatasetPath(
        user_id=userId[0],
        screen_id=str(screenId[0]),
        camera_id=str(cameraId[0]),
        monitor_id=str(monitorId[0]),
        place_id=str(placeId[0]),
        base_path=folder,
    )
    myfolder = Path(dataset_path.full_path)
    myfolder.mkdir(parents=True, exist_ok=True)
    start_time = samples["time"][0]
    fname = find_free_name(str(myfolder), str(start_time), extension=".npz")
    np.savez_compressed(fname, **samples)


def splitByID(samples: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Split samples by userId/screenId/cameraId/monitorId/placeId into separate chunks.

    Groups samples that share the same user, screen, camera, monitor, and place IDs and returns
    each group as a separate dictionary with stacked arrays.

    Args:
        samples: Dictionary of sample arrays with userId, screenId, cameraId, monitorId, placeId.

    Returns:
        List of sample dictionaries, one per unique user/screen/camera/monitor/place combination.

    Example:
        >>> chunks = splitByID(mixed_samples)
        >>> assert len(chunks) <= len(mixed_samples)
    """
    res_dict: Dict[str, List[Dict[str, np.ndarray]]] = {}
    keys = list(samples.keys())
    N = len(samples["time"])
    for i in range(N):
        sample = {k: samples[k][i] for k in keys}  # copy
        userId = sample["userId"]
        screenId = sample["screenId"]
        cameraId = sample["cameraId"]
        monitorId = sample["monitorId"]
        placeId = sample["placeId"]
        # Group by full path to distinguish same place across users
        dataset_path = DatasetPath(userId, screenId, cameraId, monitorId, placeId)
        key = dataset_path.full_path
        if key not in res_dict:
            res_dict[key] = []
        res_dict[key].append(sample)

    # transpose them to the make columnwise
    lst = list(res_dict.values())
    res: List[Dict[str, np.ndarray]] = []
    for values in lst:
        newValues: Dict[str, np.ndarray] = {}
        for k in keys:
            newValues[k] = np.array([sample[k] for sample in values])
        res.append(newValues)

    return res


def fetch(cache: Optional[str] = None) -> Callable[[str], Tuple[IO.BytesIO, bool]]:
    """Create a fetcher function that downloads or caches files.

    Returns a function that fetches URLs either directly from server or from cache.
    If cache is provided, files are cached locally to avoid repeated downloads.

    Args:
        cache: Optional path to cache directory. If None, always fetches from server.

    Returns:
        Function that takes URL and returns (BytesIO content, is_cached flag).

    Example:
        >>> fetcher = fetch(cache="./cache")
        >>> content, is_cached = fetcher("http://example.com/data.gz")
    """

    def fromServer(url: str) -> Tuple[IO.BytesIO, bool]:
        """Fetch file directly from server.

        Args:
            url: URL to fetch from.

        Returns:
            Tuple of (BytesIO content, is_cached flag).

        Raises:
            requests.RequestException: If HTTP request fails.
        """
        response = requests.get(url=url)
        response.raise_for_status()  # Raise exception for HTTP errors
        return IO.BytesIO(response.content), False

    def cached(url: str) -> Tuple[IO.BytesIO, bool]:
        """Fetch file from cache or download and cache.

        Args:
            url: URL to fetch (filename extracted for cache lookup).

        Returns:
            Tuple of (BytesIO content, is_cached flag).

        Raises:
            FileNotFoundError: If cache file does not exist and download fails.
            IOError: If cache file cannot be written.
        """
        name = Path(url).name
        cache_file = Path(cache or ".") / name
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return IO.BytesIO(f.read()), True

        response, _ = fromServer(url)
        # Read content from response. Note: BytesIO from network response is read-only,
        # so we must consume it here and create a new seekable BytesIO for return value
        content = response.read()
        with open(cache_file, "wb") as f:
            f.write(content)
        return IO.BytesIO(content), False

    if cache is not None:
        return cached
    return fromServer


def process_files(
    file_paths: List[str], fetcher: Callable[[str], Tuple[IO.BytesIO, bool]]
) -> None:
    """Process a list of files and save them organized by userId/screenId/cameraId/placeId.

    Handles deserialization errors and gzip failures gracefully, logging and continuing
    with next file. Expected failures (corrupt files) are logged and skipped.
    Unexpected IO errors (disk full, permissions) propagate with full stack trace.

    Args:
        file_paths: List of file paths to process.
        fetcher: Function that fetches file content given a path.

    Raises:
        IOError: For unexpected IO errors during file write operations (disk full, permissions).
    """
    N = len(file_paths)
    L = len(str(N))
    logger.info(f"Found {N} files to process")
    for i, file in enumerate(file_paths):
        try:
            content, isCached = fetcher(file)
        except IOError:
            # Unexpected IO error during fetch - propagate immediately
            raise
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to fetch file {file}: {e}")
            continue

        try:
            # read first file in the gz archive
            with gzip.open(content, "rb") as f:
                first_file = f.read()
        except gzip.BadGzipFile as e:
            logger.warning(f"Invalid gzip file {file}: {e}")
            continue

        try:
            samples = deserialize(first_file)
        except ValueError as e:
            logger.warning(f"Failed to deserialize file {file}: {e}")
            continue
        except AssertionError as e:
            logger.warning(f"Data validation failed for file {file}: {e}")
            continue

        src = "cache" if isCached else file
        logger.info(
            f"[{i:0{L}d}/{N:0{L}d}] Read {len(samples['time'])} samples from {src}"
        )

        # don't want to mess up with such cases
        userId = np.unique(samples["userId"])
        screenId = np.unique(samples["screenId"])
        cameraId = np.unique(samples["cameraId"])
        placeId = np.unique(samples["placeId"])

        chunks = [samples]
        needSplit = (
            (1 < len(userId))
            or (1 < len(screenId))
            or (1 < len(cameraId))
            or (1 < len(placeId))
        )
        if needSplit:
            chunks = splitByID(samples)

        for chunk in chunks:
            saveChunk(chunk, folder=str(folder / "remote"))


def get_cached_files(cache_dir: str) -> List[str]:
    """Get list of cached files from local cache directory.

    Recursively finds all .gz files in the cache directory.

    Args:
        cache_dir: Path to cache directory.

    Returns:
        List of file paths to cached .gz files.

    Example:
        >>> files = get_cached_files("cache/")
        >>> assert all(f.endswith(".gz") for f in files)
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []

    cached_files = list(cache_path.rglob("*.gz"))
    file_paths = [str(f) for f in cached_files]
    logger.info(f"Found {len(file_paths)} cached files in {cache_dir}")
    return file_paths


def main(args: argparse.Namespace) -> None:
    """Download and process samples from remote server or local cache.

    If URL is provided, fetches gzip-compressed data files from remote server.
    If URL is None, processes files from local cache directory (Data/remote-cache).
    Deserializes samples, validates metadata consistency, and saves organized
    by userId/screenId/cameraId/placeId.

    Args:
        args: Command-line arguments with 'url' (optional) and 'cache' (optional).

    Returns:
        None (writes dataset files organized by hierarchy).

    Raises:
        IOError: If files cannot be written to disk.
        ValueError: If data validation fails.
    """
    # Clear the folder
    shutil.rmtree(folder / "remote", ignore_errors=True)
    cache_dir = args.cache or str(folder / "remote-cache")
    os.makedirs(cache_dir, exist_ok=True)

    if args.url is not None:
        # Fetch from remote server
        urls = requests.get(args.url).json()
        fetcher = fetch(cache_dir)
        process_files(urls, fetcher)
    else:
        # Process local cached files
        file_paths = get_cached_files(cache_dir)

        if not file_paths:
            logger.error(f"No cached files found in {cache_dir}")
            return

        # Create a simple fetcher for local files
        def local_fetcher(file_path: str) -> Tuple[IO.BytesIO, bool]:
            """Load file from local filesystem.

            Args:
                file_path: Path to local file.

            Returns:
                Tuple of (BytesIO content, is_cached flag = True for local files).

            Raises:
                FileNotFoundError: If file does not exist.
                IOError: If read operation fails.
            """
            with open(file_path, "rb") as f:
                return IO.BytesIO(f.read()), True

        process_files(file_paths, local_fetcher)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process remote data files from server or local cache"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Full URL to the list.php on remote server. If not provided, uses local cache.",
        default=None,
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Path to the cache folder (default: Data/remote-cache)",
        default=None,
    )

    args = parser.parse_args()
    main(args)
