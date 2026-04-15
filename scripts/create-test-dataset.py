#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from collections import defaultdict
from Core.data.DataSampler import DataSampler
from Core.data.SamplesStorage import SamplesStorage
import Core.Utils as Utils
from Core.Utils import DatasetInfo
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Iterator, Any
from Core.logging_config import get_logger

ROOT_FOLDER = Path(__file__).parent.parent
logger = get_logger(__name__)

# Default batch size constant (can be overridden via command line)
DEFAULT_BATCH_SIZE = 512

# Output formatting constants
ONE_MB = 1024 * 1024  # Bytes per megabyte for file size display


def samplesStream(
    params: Dict[str, Any],
    filename: str,
    info: DatasetInfo,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Dict[str, np.ndarray]]:
    """Stream augmented samples from dataset file.

    Args:
        params: Sampling parameters (timesteps, augmentation settings).
        filename: Path to dataset npz file.
        info: Dataset information (userId, screenId, cameraId, monitorId, placeId indices).
        batch_size: Batch size for streaming samples (default: DEFAULT_BATCH_SIZE).

    Yields:
        Dictionary of sample data with keys like 'X_points', 'Y_points', etc.

    Example:
        >>> stream = samplesStream(params, "data.npz", info, 32)
        >>> sample = next(stream)
        >>> assert "X_points" in sample
    """
    # Get numeric indices from info (screenId uses composite key for uniqueness)
    ds = DataSampler(
        SamplesStorage(
            userId=info.user_idx,
            screenId=info.screen_idx,
            cameraId=info.camera_idx,
            monitorId=info.monitor_idx,
            placeId=info.place_idx,
        ),
        defaults=params,
        batch_size=batch_size,
        minFrames=params["timesteps"],
    )
    dataset = Utils.dataset_from(filename)
    if dataset is not None:
        ds.addBlock(dataset)

    N = ds.totalSamples
    for i in range(0, N, batch_size):
        indices = list(range(i, min(i + batch_size, N)))
        # sampleByIds returns (batch, rejected_indices, accepted_indices)
        # Only batch is used here; unused indices are discarded with underscore
        batch, _, _ = ds.sampleByIds(ids=indices)
        if batch is None:
            continue

        (x_dict, y_dict), batch_size_actual = batch

        for idx in range(batch_size_actual):
            res = {}
            # Extract X
            for k, v in x_dict["clean"].items():
                item = v[idx, None]
                res[f"X_{k}"] = Utils.to_numpy(item)

            # Extract Y
            for k, v in y_dict.items():
                item = v[idx, None]
                res[f"Y_{k}"] = Utils.to_numpy(item)

            yield res


def batches(
    params: Dict[str, Any],
    filename: str,
    info: DatasetInfo,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Dict[str, np.ndarray]]:
    """Batch samples into groups of specified batch size.

    Args:
        params: Sampling parameters (timesteps, augmentation settings).
        filename: Path to dataset npz file.
        info: Dataset information (placeId, userId, screenId indices).
        batch_size: Target batch size for output batches (default: DEFAULT_BATCH_SIZE).

    Yields:
        Dictionaries of batched sample data, padded to batch_size if needed.

    Example:
        >>> stream = batches(params, "data.npz", info, 64)
        >>> batch = next(stream)
        >>> assert batch["Y_result"].shape[0] == 64
    """
    data: Dict[str, List[np.ndarray]] = defaultdict(list)
    for sample in samplesStream(
        params=params, filename=filename, info=info, batch_size=batch_size
    ):
        for k, v in sample.items():
            data[k].append(v)

        if batch_size <= len(data["Y_result"]):
            # Convert list of arrays to concatenated array
            batch_data: Dict[str, np.ndarray] = {
                k: np.concatenate(v, axis=0) for k, v in data.items()
            }
            yield batch_data
            data = defaultdict(list)

    if 0 < len(data["Y_result"]):
        # copy data to match batch size
        padded_data: Dict[str, List[Any]] = {}
        for k, v_items in data.items():
            # v_items is List[np.ndarray] from defaultdict(list)
            arr_list: List[np.ndarray] = list(v_items)
            while len(arr_list) < batch_size:
                arr_list.extend(arr_list)
            padded_data[k] = arr_list[:batch_size]
        batch_data = {
            k: np.concatenate(arr_list, axis=0) for k, arr_list in padded_data.items()
        }
        yield batch_data


############################################


def generateTestDataset(
    params: Dict[str, Any],
    filename: str,
    outputFolder: str,
    info: DatasetInfo,
    train_index: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Generate test dataset files from samples.

    Args:
        params: Sampling parameters.
        filename: Path to input dataset file.
        outputFolder: Output directory for test files.
        info: Dataset information (placeId, userId, screenId indices).
        train_index: Starting index for file naming.
        batch_size: Batch size for sample streaming (default: DEFAULT_BATCH_SIZE).

    Returns:
        Updated train_index after processing all batches.

    Raises:
        OSError: If output folder cannot be created or files cannot be written.

    Example:
        >>> index = generateTestDataset(params, "data.npz", "output", info, 0)
        >>> assert index > 0
    """
    # generate test dataset
    totalSize = 0
    output_path = Path(outputFolder)
    output_path.mkdir(parents=True, exist_ok=True)
    for bIndex, batch in enumerate(
        batches(
            params=params,
            filename=filename,
            info=info,
            batch_size=batch_size,
        )
    ):
        fname = output_path / f"test-{train_index}.npz"
        np.savez_compressed(str(fname), **batch)
        shapes_info = {k: v.shape for k, v in batch.items()}
        logger.debug(f"Batch shapes: {shapes_info}")
        # get fname size
        size = fname.stat().st_size
        totalSize += size
        logger.info(
            f"{bIndex + 1} | Size: {size / ONE_MB:.1f} MB | Total: {totalSize / ONE_MB:.1f} MB"
        )
        train_index += 1
    logger.info("Done")
    return train_index


def main(args: argparse.Namespace) -> None:
    """Generate test datasets from remote data with specified parameters.

    Args:
        args: Command-line arguments with steps, batch_size, and output folder.

    Returns:
        None (writes test dataset files to disk).

    Raises:
        FileNotFoundError: If dataset files do not exist.
        ValueError: If stats.json is invalid.
    """
    PARAMS = [
        dict(timesteps=args.steps, stepsSampling="last"),
        # Other parameter sets can be added here in the future
    ]
    folder = ROOT_FOLDER / "Data" / "remote"
    json_path = str(folder / "stats.json")

    # remove all content from the output folder
    output_path = Path(args.output)
    shutil.rmtree(output_path, ignore_errors=True)

    # Use dataset_from_stats to iterate through valid datasets (respects blacklist)
    dataset_list = list(Utils.dataset_from_stats(json_path))
    logger.info("Found test files: %d", len(dataset_list))
    train_index = 0
    for idx, dataset_info in enumerate(dataset_list):
        testFile = Path(dataset_info.path.full_path) / "test.npz"
        if not testFile.exists():
            continue
        logger.info("Processing %s", testFile)
        for params in PARAMS:
            targetFolder = output_path / f"test-{idx}"
            train_index = generateTestDataset(
                params,
                str(testFile),
                outputFolder=str(targetFolder),
                info=dataset_info,
                train_index=train_index,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5, help="Number of timesteps")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size of the test dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder",
        default=str(ROOT_FOLDER / "Data" / "test-main"),
    )
    args = parser.parse_args()
    main(args)
