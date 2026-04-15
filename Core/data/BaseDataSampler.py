"""Base data sampler for managing frame trajectories and sampling."""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from math import ceil
from Core.logging_config import get_logger
from Core.Constants import DATA_SAMPLER_MAX_RETRIES

from Core.data.SamplerInterface import SamplerInterface
from Core.data.FilteredDataset import FilteredDataset


logger = get_logger(__name__)


class BaseDataSampler(SamplerInterface):
    """Abstract base class for data sampling with temporal trajectory management.

    Manages sampling strategies over a filtered dataset with temporal information
    and provides methods for batch sampling, trajectory extraction, and frame
    selection. Handles time window filtering and frame count validation.

    Attributes:
        _dataset: FilteredDataset instance managing storage and valid samples
        _defaults: Default sampling parameters
        _batchSize: Number of samples per batch
        _cumulative_time: Whether time values are cumulative or deltas
    """

    def __init__(
        self,
        storage: Any,
        batch_size: int,
        minFrames: int,
        defaults: Optional[Dict[str, Any]] = None,
        maxT: float = 1.0,
        cumulative_time: bool = True,
    ) -> None:
        """Initialize base data sampler.

        Args:
            storage: Storage object containing sample data. Must support:
                - __getitem__(idx) to access sample at index
                - __len__() for total samples
                - add(sample) to add single sample
                - addBlock(samples) to add multiple samples
            batch_size: Number of samples per batch for training
            minFrames: Minimum number of frames in a valid trajectory
            defaults: Default parameters for sampling (default: {})
            maxT: Maximum time window for selecting frames (default: 1.0 second)
            cumulative_time: If True, time is cumulative; if False, time contains
                deltas between frames (default: True)

        Raises:
            ValueError: If batch_size or minFrames <= 0
            TypeError: If storage doesn't support required methods
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._dataset = FilteredDataset(storage, minFrames, maxT)
        self._defaults = defaults if defaults is not None else {}
        self._batchSize = batch_size
        self._cumulative_time = cumulative_time

        logger.info(
            "Initialized %s with batch_size=%d, minFrames=%d, maxT=%.2f, cumulative_time=%s",
            self.__class__.__name__,
            batch_size,
            minFrames,
            maxT,
            cumulative_time,
        )

    def reset(self) -> None:
        """Shuffle samples and reset iteration pointer.

        Called at the beginning of each epoch to randomize sample order.
        """
        self._dataset.reset()
        logger.debug("Reset sampler: shuffled %d samples", len(self._dataset))

    def __len__(self) -> int:
        """Get number of batches.

        Returns:
            Number of complete batches that can be formed from samples
        """
        return ceil(len(self._dataset) / self._batchSize)

    def add(self, sample: Dict[str, Any]) -> int:
        """Add single sample to storage.

        Args:
            sample: Sample dictionary containing at minimum:
                - 'time': Timestamp for temporal ordering
                - Other fields specific to the data type

        Returns:
            Index of added sample in storage
        """
        return self._dataset.add_sample(sample)

    def addBlock(self, samples: Any) -> None:
        """Add multiple samples to storage in batch.

        Args:
            samples: Samples in bulk format. Accepts either:
                - List of sample dictionaries: [{'time': 0.1, ...}, ...]
                - Batched dict with array values: {'time': [0.1, 0.2], ...}
        """
        self._dataset.add_samples_block(samples)

    def _trajectory(self, mainInd: int) -> List[int]:
        """Get past frame indices for a sample.

        Returns indices of frames before the main sample within the time window.

        Args:
            mainInd: Reference sample index

        Returns:
            List of frame indices from trajectory start to main sample (exclusive)
        """
        minInd = self._dataset.trajectory_start(mainInd)
        return list(range(minInd, mainInd))

    def _prepareT(self, res: List[int]) -> Optional[np.ndarray]:
        """Prepare time array for frame indices.

        Validates that frames have strictly increasing times and creates
        time array (either cumulative or deltas).

        Args:
            res: List of frame indices

        Returns:
            Time array with length == len(res), or None if time is invalid

        Raises:
            AssertionError: If time array length mismatch
        """
        T = np.array([self._dataset.storage[ind]["time"] for ind in res])
        T -= T[0]
        diff = np.diff(T, 1)
        idx = np.nonzero(diff)[0]

        if len(diff) == len(idx):  # all ok
            result = T
            if not self._cumulative_time:
                result = np.insert(diff, 0, 0.0)
            assert len(result) == len(
                T
            ), f"Time array length {len(T)} != frames {len(result)}"
            return result
        return None  # Time is not consistent

    def _reshapeSteps(
        self, values: Tuple[np.ndarray, ...], steps: Optional[int]
    ) -> Tuple[np.ndarray, ...]:
        """Reshape frame arrays to add timestep dimension.

        Takes (B*steps, ...) shaped arrays and reshapes to (B, steps, ...).

        Args:
            values: Tuple of arrays with batch dimension
            steps: Number of timesteps (if None, returns unchanged)

        Returns:
            Tuple of reshaped arrays
        """
        if steps is None:
            return values

        res = []
        for x in values:
            B, *s = x.shape
            newShape = (B // steps, steps, *s)
            res.append(x.reshape(newShape))

        return tuple(res)

    @property
    def totalSamples(self) -> int:
        """Get total number of samples in storage.

        Returns:
            Total sample count
        """
        return self._dataset.total_samples

    def validSamples(self) -> List[int]:
        """Get sorted list of valid sample indices.

        Returns indices of samples that meet minimum frame requirements.

        Returns:
            Sorted list of valid sample indices
        """
        return self._dataset.valid_indices()

    def _stepsFor(
        self,
        mainInd: int,
        steps: Optional[int] = None,
        stepsSampling: str = "uniform",
        **kwargs: Any,
    ) -> Optional[List[Tuple[int, float]]]:  # type: ignore[unused-argument]
        """Get frame indices and times for given number of timesteps.

        Samples 'steps' frames from trajectory with corresponding times.
        Returns None if sampling fails (not enough frames or invalid times).

        Args:
            mainInd: Main/reference frame index
            steps: Number of frames to sample (None or 1 returns just main frame)
            stepsSampling: Sampling strategy ('uniform', 'uniform time', etc.)
            **kwargs: Additional sampling parameters (kept for API compatibility)

        Returns:
            List of (frame_index, time) tuples, or None if sampling failed
        """
        if (steps is None) or (steps == 1):
            return [(mainInd, 0.0)]

        if mainInd < steps:
            return None

        samples = self._trajectory(mainInd)
        if len(samples) < (steps - 1):
            return None

        # Try to sample valid frames (up to DATA_SAMPLER_MAX_RETRIES attempts)
        for _ in range(DATA_SAMPLER_MAX_RETRIES):
            res = self.framesFor(mainInd, samples, steps, stepsSampling)
            T = self._prepareT(res)

            if T is not None:
                assert len(res) == len(
                    T
                ), f"Frame/time mismatch: {len(res)} != {len(T)}"
                return [tuple(x) for x in zip(res, T)]

        return None

    def _sampleSteps(
        self, retries: int, timesteps: Optional[int], kwargs: Dict[str, Any]
    ) -> Optional[List[Tuple[int, float]]]:
        """Sample steps with retry logic.

        Attempts to sample valid timesteps from the next available sample.
        On failure after retries, removes the problematic sample from the
        valid samples list.

        Args:
            retries: Maximum number of retry attempts
            timesteps: Number of timesteps to sample
            kwargs: Additional sampling parameters

        Returns:
            List of (frame_index, time) tuples, or None if all retries failed
        """
        idx = self._dataset.next_sample_index()
        for _ in range(retries):
            sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
            if sampledSteps is not None:
                return sampledSteps
        self._dataset.remove_sample(idx)
        return None

    def _indexes2XY(  # type: ignore[misc] # Abstract method signature varies by subclass implementation
        self, indexesAndTime: List[Tuple[int, float]], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Dict, Dict], int]:
        """Convert frame indices to input/output tensors (abstract).

        Must be implemented by subclasses to handle sampler-specific data conversion.

        Args:
            indexesAndTime: List of (frame_index, time) tuples
            kwargs: Sampling parameters

        Returns:
            Tuple of (X, Y), N data
        """
        raise NotImplementedError("Subclasses must implement _indexes2XY()")

    def sampleByIds(
        self, ids: List[int], **kwargs: Any
    ) -> Tuple[Optional[Tuple[Any, Any]], List[int], List[int]]:
        """Sample sequences for multiple frame indices.

        Args:
            ids: List of frame indices to sample
            **kwargs: Sampling parameters

        Returns:
            Tuple of (result, rejected_ids, accepted_ids) where:
            - result: (X, Y) tuple or None if no samples produced
            - rejected_ids: Indices that failed sampling
            - accepted_ids: Indices that succeeded
        """
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get("timesteps")
        sampledSteps = []
        rejected = []
        accepted = []

        for idx in ids:
            sample = self._stepsFor(idx, steps=timesteps, **kwargs)
            if sample is None:
                rejected.append(idx)
                continue

            accepted.append(idx)
            sampledSteps.extend(sample)

        result = None
        if len(sampledSteps) > 0:
            result = self._indexes2XY(sampledSteps, kwargs)

        return result, rejected, accepted

    def framesFor(
        self, mainInd: int, samples: List[int], steps: int, stepsSampling: Any
    ) -> List[int]:
        """Select frame indices using various sampling strategies.

        Selects 'steps' number of frames from candidates using the specified
        sampling strategy. Ensures main frame is always included.

        Args:
            mainInd: Index of main/reference frame
            samples: Available frame candidates to sample from
            steps: Number of frames to select
            stepsSampling: Sampling strategy - can be:
                - 'uniform time': Select frames uniformly across time window
                - 'uniform': Randomly select frames uniformly
                - 'last': Select most recent frames
                - dict: Advanced sampling with 'max frames' parameter

        Returns:
            Sorted list of selected frame indices (length == steps)

        Raises:
            ValueError: If sampling strategy is unknown
            AssertionError: If not enough samples for requested steps
        """
        from Core.data.sampling_strategies import (
            uniform_time_sampling,
            uniform_sampling,
            last_sampling,
            dict_sampling,
        )

        # Dispatch to appropriate sampling strategy
        strategies: Dict[str, Callable[..., List[int]]] = {
            "uniform time": uniform_time_sampling,
            "uniform": uniform_sampling,
            "last": last_sampling,
            "dict": dict_sampling,
        }
        kwargs = {
            "mainInd": mainInd,
            "sampling": stepsSampling,
            "steps": steps,
            "samples": samples,
            "storage": self._dataset.storage,
        }
        key = stepsSampling
        # Determine strategy and parameters
        if isinstance(stepsSampling, dict):
            key = "dict"

        if key not in strategies:
            raise ValueError(f"Unknown sampling method: {stepsSampling}")

        # Call strategy and combine with mainInd
        strategy: Callable[..., List[int]] = strategies[key]
        selected = strategy(**kwargs)
        res = list(sorted(selected + [mainInd]))
        assert len(res) == steps, f"Expected {steps} samples, got {len(res)}"
        assert len(res) == len(set(res)), f"Result has duplicate frame indices: {res}"
        return res
