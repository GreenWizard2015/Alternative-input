"""Custom data loader for filter training from NPZ files.

Loads filter datasets from NPZ files containing left eye, right eye images and
is_valid labels for binary classification task.
"""

from typing import Dict, Tuple, Generator, Any, List
import tensorflow as tf
import numpy as np
import os
from math import ceil
from Core.logging_config import get_logger

logger = get_logger(__name__)


class BalancedSampler:
    def __init__(self, labels):
        self._samples = {}
        uniq = np.unique(labels)
        self._labels = uniq
        for label in uniq:
            idx = np.where(label == labels)[0]
            self._samples[label] = idx
        self.reset()

    def sample(self):
        current_label = self._labels[self._current_label]
        current_samples = self._samples[current_label]
        res = current_samples[self._current_idx % len(current_samples)]
        self._current_label += 1
        if len(self._labels) <= self._current_label:
            self._current_label = 0
            self._current_idx += 1
        return res

    def reset(self):
        for v in self._samples.values():
            np.random.shuffle(v)
        np.random.shuffle(self._labels)
        self._current_label = 0
        self._current_idx = 0


class FilterDataLoader:
    """Sequence loader for filter training data from NPZ files.

    Loads training data from NPZ files containing eye images and binary labels.
    Simplified implementation without BaseDataSampler inheritance.

    Attributes:
        _npz_file: Path to the specific filter-train.npz file
        _data_cache: Cache for loaded data to avoid repeated disk I/O
        _batch_size: Batch size for training
        _defaults: Default sampling parameters with augmentation settings
        _samples: List of sample indices for sampling
        _current_epoch: Current epoch counter for reset functionality
    """

    def __init__(
        self,
        npz_file: str,
        batch_size: int = 32,
        train: bool = False,
        batch_per_epoch=-1,
    ) -> None:
        """Initialize filter data loader.

        Args:
            npz_file: Path to specific filter-train.npz file
            batch_size: Batch size for training
        """
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Training file not found: {npz_file}")

        self._batch_size = batch_size
        self._defaults = {}
        self._data_cache: Dict[str, np.ndarray] = dict(np.load(npz_file))
        self._data_cache["is_valid"] = self._data_cache["is_valid"].astype(np.float32)
        self._samples_count = len(self._data_cache["is_valid"])
        self._train = train

        if 0 < batch_per_epoch:
            self._samples_count = max(self._samples_count, batch_per_epoch * batch_size)

        if train:
            self._sampler = BalancedSampler(self._data_cache["is_valid"])
        self.reset()

    def __len__(self) -> int:
        """Get number of batches per epoch for Keras compatibility."""
        return ceil(self._samples_count / self._batch_size)

    def sample(self, _=None, **kwargs: Any) -> Generator[Tuple[Dict, Dict], None, None]:
        """Sample a batch of single frames.

        Args:
            batch_id: Optional batch ID for compatibility with training_utils
            **kwargs: Sampling parameters with augmentation settings

        Returns:
            Tuple of (X_dict, Y_dict) or None if no samples
        """
        kwargs = {**self._defaults, **kwargs}
        N = kwargs.get("N", self._batch_size)

        # Sample N indices randomly (or all if fewer than N)
        if self._train:
            sampled_indices = (
                np.array([self._sampler.sample() for _ in range(N)])
                % self._samples_count
            )
        else:
            sampled_indices = (
                np.array([self._current_sample + idx for idx in range(N)])
                % self._samples_count
            )
            self._current_sample += N

        XY, _ = self._indexes2XY(list(sampled_indices), kwargs)
        yield XY

    def _indexes2XY(
        self, indexes: List[int], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Dict, Dict], int]:
        """Convert frame indices to input/output tensors.

        Args:
            indexes: List of idx
            kwargs: Parameters including augmentation settings

        Returns:
            Tuple of ((X_dict, Y_dict), actual_size)
        """
        # Prepare target data (is_valid labels)
        y = self._data_cache["is_valid"][indexes]

        # Extract eye images and add channel dimension (48, 48) -> (48, 48, 1)
        left_eye_images = self._data_cache["left eye"][indexes]
        right_eye_images = self._data_cache["right eye"][indexes]

        # Add channel dimension for TensorFlow compatibility
        left_eye_images = left_eye_images[..., np.newaxis].astype(np.float32) / 255.0
        right_eye_images = right_eye_images[..., np.newaxis].astype(np.float32) / 255.0

        if self._train:
            left_eye_images, right_eye_images = self._augm(
                left_eye_images, right_eye_images, y
            )

        # Create tensor structure directly
        X = {
            "left eye": tf.constant(left_eye_images, dtype=tf.float32),
            "right eye": tf.constant(right_eye_images, dtype=tf.float32),
        }

        if self._train:
            X = {
                "clean": X,
                "augmented": X,
            }

        # Convert target to tensor
        Y = {"is_valid": tf.constant(y, dtype=tf.float32)}

        return ((X, Y), len(y))

    def reset(self):
        if self._train:
            self._sampler.reset()
        else:
            self._current_sample = 0

    def on_epoch_start(self):
        """Called at the start of each training epoch."""
        self.reset()

    def on_epoch_end(self):
        """Called at the end of each training epoch."""
        pass

    def _augm(self, left_eye_images, right_eye_images, y):
        labels = np.unique(y)
        for label in labels:
            idx = np.where(label == y)[0]
            N = len(idx)
            all_samples = np.concatenate(
                [left_eye_images[idx], right_eye_images[idx]], axis=0
            )
            idx2 = np.arange(len(all_samples))
            np.random.shuffle(idx2)
            left_eye_images[idx] = all_samples[idx2[:N]]
            right_eye_images[idx] = all_samples[idx2[N:]]

        return left_eye_images, right_eye_images
