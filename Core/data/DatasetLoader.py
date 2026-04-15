"""Dataset loader that combines multiple samplers with configurable sampling strategies."""

from typing import Any, Dict, List, Tuple, Type
import Core.Utils as Utils
import os
from Core.data.SamplesStorage import SamplesStorage
import numpy as np
from Core.logging_config import get_logger
from queue import Queue
from threading import Thread
import math
import time

logger = get_logger(__name__)


class DatasetLoader:
    """Multi-dataset loader with round-robin batch generation.

    Loads multiple datasets from folders and generates batches infinitely
    using round-robin selection with shuffled dataset order per epoch.

    Attributes:
        _datasets: List of sampler instances for each dataset
        _dataset_index: Round-robin indices for dataset selection
        _dataset_index_pos: Current position in dataset index
        _batchSize: Default batch size for sampling
        batchPerEpoch: Number of batches per epoch
        _sampling_mode: Strategy for epoch length determination
    """

    SAMPLING_MODES = {"oversample", "undersample"}

    # Formula: dataset_size × num_datasets / batch_size
    # (Accounts for per-batch fair allocation where each dataset gets batch_size/num_datasets samples)
    _MODE_ESTIMATION = {
        "oversample": lambda dataset_sizes, batch_size, num_datasets: (
            (max(dataset_sizes) * num_datasets) // batch_size
        ),
        "undersample": lambda dataset_sizes, batch_size, num_datasets: (
            (min(dataset_sizes) * num_datasets) // batch_size
        ),
    }

    def __init__(
        self,
        json_path: str,
        samplerArgs: Dict[str, Any],
        sampler_class: Type,
        batchPerEpoch: int,
        minSubBatchSize: int = 16,
        sampling_mode: str = "oversample",
    ) -> None:
        """Initialize dataset loader with multiple samplers and sampling strategy.

        Loads datasets from JSON config and sets up batch generation with specified
        sampling mode. Each mode controls epoch length while maintaining per-batch
        fair allocation across all datasets.

        Args:
            json_path: Path to statistics JSON file containing dataset configuration.
            samplerArgs: Arguments to pass to sampler constructor (batch_size, etc).
            sampler_class: Sampler class to instantiate for each dataset.
            batchPerEpoch: Number of batches per epoch. Use -1 for auto-estimation
                based on sampling_mode and dataset sizes.
            minSubBatchSize: Minimal number of samples from each dataset.
            sampling_mode: Strategy for epoch length determination (default: "oversample").
                - "oversample": Extends epochs until largest dataset exhausted. Small datasets
                  heavily resampled to match largest. Epoch length: max(sizes) × num_datasets / batch_size.
                  Use when all data is equally important and you want maximum coverage per epoch.
                - "undersample": Stops when smallest dataset exhausted. Large datasets only
                  partially used each epoch. Epoch length: min(sizes) × num_datasets / batch_size.
                  Use when respecting scarce datasets is important.

        Raises:
            FileNotFoundError: If JSON config or dataset files not found.
            ValueError: If sampling_mode not in {"oversample", "undersample"}.
        """
        if sampling_mode not in self.SAMPLING_MODES:
            raise ValueError(
                f"Invalid sampling_mode '{sampling_mode}'. "
                f"Must be one of {self.SAMPLING_MODES}"
            )
        self._sampling_mode = sampling_mode
        self._datasets: List[Any] = []
        for dataset_info in Utils.dataset_from_stats(json_path):
            dataset = os.path.join(dataset_info.path.full_path, "train.npz")
            if not os.path.exists(dataset):
                continue
            logger.info(
                f"ID: {(dataset_info.user_idx, dataset_info.screen_idx, dataset_info.camera_idx, dataset_info.monitor_idx, dataset_info.place_idx)}. Index: {1 + len(self._datasets)}"
            )
            ds = sampler_class(
                SamplesStorage(
                    userId=dataset_info.user_idx,
                    screenId=dataset_info.screen_idx,
                    cameraId=dataset_info.camera_idx,
                    monitorId=dataset_info.monitor_idx,
                    placeId=dataset_info.place_idx,
                ),
                **samplerArgs,
            )
            ds.addBlock(Utils.dataset_from(dataset))
            if 0 < len(ds.validSamples()):
                self._datasets.append(ds)

        if 0 == len(self._datasets):
            raise FileNotFoundError(f'No training dataset found in "{json_path}"')

        validSamples = {
            i: len(ds.validSamples()) for i, ds in enumerate(self._datasets)
        }

        logger.info(
            f"Loaded {len(self._datasets)} datasets with {sum(validSamples.values())} valid samples"
        )

        self._dataset_index = np.arange(len(self._datasets) * 2) % len(self._datasets)
        self._dataset_index_pos = 0  # Track current position in round-robin

        self._batchSize: int = samplerArgs.get("batch_size", 16)
        self._minSubBatchSize: int = minSubBatchSize

        if batchPerEpoch == -1:
            batchPerEpoch = self._estimate_batches_per_epoch(validSamples)
            logger.info(
                f"Auto-estimated batchPerEpoch={batchPerEpoch} for mode '{sampling_mode}' "
                f"(dataset sizes: {list(validSamples.values())})"
            )

        self.batchPerEpoch: int = batchPerEpoch
        logger.info(
            f"Sampling mode: {sampling_mode} | batchPerEpoch={self.batchPerEpoch}"
        )

        self._reset()

        # Start batch generation thread (runs indefinitely in background)
        self._batch_queue: Queue = Queue(maxsize=3)
        self._generator_thread: Thread = Thread(
            target=self._generate_batches, daemon=True
        )
        self._generator_thread.start()

    def _estimate_batches_per_epoch(self, validSamples: Dict[int, int]) -> int:
        """Estimate batches per epoch based on sampling mode and dataset sizes.

        Uses mode-specific estimation formula from _MODE_ESTIMATION.
        Formula: dataset_size × num_datasets / batch_size

        Args:
            validSamples: Dict mapping dataset index to valid sample count

        Returns:
            Estimated number of batches per epoch (minimum 1)

        Raises:
            ValueError: If sampling mode not found in _MODE_ESTIMATION
        """
        if not validSamples:
            return 1

        estimation_fn = self._MODE_ESTIMATION.get(self._sampling_mode)
        if estimation_fn is None:
            raise ValueError(f"No estimation function for mode '{self._sampling_mode}'")

        estimate = estimation_fn(
            validSamples.values(), self._batchSize, len(self._datasets)
        )
        return 1 + estimate

    def _generate_batches(self) -> None:
        """Generate batches in background thread indefinitely."""
        try:
            while True:
                # start_time = time.time()
                batch = self._sample_batch()
                # logger.info(f"Batch created in {((time.time() - start_time) * 1000):.2f}ms")
                self._batch_queue.put(batch, timeout=None)
        except (IOError, ValueError, RuntimeError) as e:
            logger.error(f"Error generating batches: {type(e).__name__}: {e}")
            self._batch_queue.put(e, timeout=None)

    def _sample_batch(self) -> Any:
        """Generate a single batch with batch pre-allocation.

        Internal method called by generator thread to create batches using
        round-robin dataset selection and pre-allocated batch structure.

        Returns:
            Merged batch from sampled datasets.
        """
        batchSize = self._batchSize
        first_dataset = self._datasets[0]

        # Pre-allocate batch structure
        batch = first_dataset.create_empty_batch(batchSize)
        batch_index = 0
        while batch_index < batchSize:
            datasetIds, counts = self._getBatchStats(
                batchSize - batch_index,
                max_samples=max(
                    self._minSubBatchSize,
                    math.ceil(batchSize / float(len(self._datasets))),
                ),
            )
            for datasetId, N in zip(datasetIds, counts):
                dataset = self._datasets[datasetId]
                # Sample directly into pre-allocated batch
                remaining = batchSize - batch_index
                if remaining <= 0:
                    continue
                samples_to_take = min(N, remaining)

                sampled = dataset.sample(
                    N=samples_to_take, batch=batch, batch_index=batch_index
                )
                if sampled is not None:
                    _, actual_count = sampled
                    batch_index += actual_count

        # Merge samples to shuffle batch
        return first_dataset.merge(samples=[batch], expected_batch_size=batch_index)

    def on_epoch_start(self) -> None:
        """Signal start of a new epoch (no-op for this loader).

        This is part of the Keras Sequence interface.
        """
        pass

    def _reset(self) -> None:
        """Reset loader state for new epoch.

        Shuffles dataset order and resets round-robin position.
        """
        np.random.shuffle(self._dataset_index)
        self._dataset_index_pos = 0

    def on_epoch_end(self) -> None:
        """Signal end of epoch (no-op for this loader).

        This is part of the Keras Sequence interface.
        """
        pass

    def __len__(self) -> int:
        """Get number of batches per epoch.

        Returns:
            Number of batches that will be yielded in one epoch.
        """
        return self.batchPerEpoch

    def _getBatchStats(
        self, batchSize: int, max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get dataset IDs and sample counts for next batch.

        Uses round-robin selection from shuffled dataset index to ensure
        fair representation across all datasets.

        Args:
            batchSize: Number of samples needed in batch.
            max_samples: Maximum samples to take from a single dataset per iteration.

        Returns:
            Tuple of (dataset_ids, sample_counts_per_dataset).
        """
        allocated: Dict[int, int] = {}
        filled = 0

        while filled < batchSize:
            # Reset position if we've gone through the entire shuffled buffer
            if self._dataset_index_pos >= len(self._dataset_index):
                self._reset()

            # Pick next dataset in round-robin order from shuffled index
            dataset_id = self._dataset_index[self._dataset_index_pos]
            self._dataset_index_pos += 1

            # Take up to max_samples from this dataset
            take = min(max_samples, batchSize - filled)
            allocated[dataset_id] = allocated.get(dataset_id, 0) + take
            filled += take

        datasetIds = np.array(sorted(allocated.keys()))
        counts = np.array([allocated[did] for did in datasetIds])
        return datasetIds, counts

    def sample(self, **kwargs: Any) -> Any:
        """Sample a batch from the pre-generated queue.

        Gets a pre-generated batch from the background thread queue.
        Waits for batch generation if queue is empty.

        Args:
            **kwargs: Sampling arguments passed to samplers (batch_size, etc).
                Must include 'batch_size' or defaults to self._batchSize.

        Returns:
            Merged batch from all sampled datasets. Returns tuple of (X, Y) where:
            - X is a dict with 'clean' and 'augmented' keys
            - Y is a tuple of output arrays

        Example:
            >>> loader = DatasetLoader(...)
            >>> batch = loader.sample(batch_size=32)
            >>> X, Y = batch
            >>> X["clean"].shape  # Input data dict
        """
        start_time = time.time()
        item = self._batch_queue.get(timeout=None)
        delta_ms = (time.time() - start_time) * 1000
        if 25 < delta_ms:
            logger.warning(f"Waited for batch {delta_ms:.2f}ms")
        # If an exception was put in the queue, re-raise it
        if isinstance(item, Exception):
            raise item
        return item
