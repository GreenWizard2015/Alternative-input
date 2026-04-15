"""Data sampler for extracting temporal frame sequences with augmentation.

This module provides sampling of N frame sequences from the dataset, where N is
the number of timesteps. Returns (X, Y) tuples with input and target data.

Data format:
    Input (X): Face points, left eye, right eye, time, user/place/screen IDs
    Output (Y): Target points for gaze/interaction prediction
"""

from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import numpy as np
import tensorflow as tf
from Core.Constants import FACEMESH_LANDMARK_COUNT
from Core.logging_config import get_logger
from Core import Utils
from Core.data import DataSampler_utils as DSUtils
from Core.data.BaseDataSampler import BaseDataSampler
from Core.data.DataValidation import validate_sample

logger = get_logger(__name__)


class DataSampler(BaseDataSampler):
    """Temporal data sampler with augmentation support.

    Samples N frames from storage where N is the number of timesteps.
    Supports various sampling strategies and applies data augmentation.

    Attributes:
        Inherits from BaseDataSampler
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
        """Initialize data sampler.

        Args:
            storage: Storage object with sample data
            batch_size: Samples per batch
            minFrames: Minimum frames per trajectory
            defaults: Default sampling parameters
            maxT: Maximum time window (seconds)
            cumulative_time: Whether time is cumulative or deltas
        """
        super().__init__(
            storage, batch_size, minFrames, defaults, maxT, cumulative_time
        )

    def sample(self, **kwargs: Any) -> Optional[Tuple[Tuple[Dict, Dict], int]]:
        """Sample a batch of frame sequences.

        Iterates through valid samples and returns a batch of (X, Y) tuples.
        If batch pre-allocation is provided, populates it instead of creating new tensors.

        Args:
            **kwargs: Sampling parameters:
                - N: Number of sequences (default: batch_size)
                - timesteps: Frames per sequence
                - batch: Pre-allocated batch tuple (X, Y) from create_empty_batch()
                - batch_index: Index within batch to write data to (requires batch parameter)
                - Augmentation params: pointsNoise, pointsDropout, eyesAdditiveNoise, etc.

        Returns:
            Tuple of ((X_dict, Y_dict), actual_size) where:
            - (X_dict, Y_dict): Data tuple with input and target data
            - actual_size: Number of sequences actually sampled
            Or None if no samples obtained
        """
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get("timesteps")
        N = kwargs.get("N", self._batchSize)
        indexes = []

        for _ in range(N):
            sampled = self._sampleSteps(retries=10, timesteps=timesteps, kwargs=kwargs)
            if sampled is not None:
                indexes.extend(sampled)

        if not indexes:
            return None

        (X, Y), B = self._indexes2XY(indexes, kwargs)
        validate_sample((X, Y))
        return ((X, Y), B)

    def sampleById(
        self, idx: int, **kwargs: Any
    ) -> Optional[Tuple[Tuple[Dict, Dict], int]]:
        """Sample a sequence starting at specific frame index.

        Args:
            idx: Frame index to use as reference
            **kwargs: Sampling parameters (timesteps, augmentation params, etc.)

        Returns:
            Tuple of ((X_dict, Y_dict), actual_size) or None if sampling failed
        """
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get("timesteps")
        sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)

        if sampledSteps is None:
            return None

        return self._indexes2XY(sampledSteps, kwargs)

    def checkById(self, idx: int, **kwargs: Any) -> bool:
        """Check if frame index can produce a valid sample.

        Args:
            idx: Frame index to check
            **kwargs: Sampling parameters

        Returns:
            True if valid sample can be produced, False otherwise
        """
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get("timesteps")
        sampledSteps = self._stepsFor(idx, steps=timesteps, **kwargs)
        return bool(sampledSteps)

    @lru_cache(None)
    def _targetFor(self, ind: int) -> np.ndarray:
        """Get target output (goal/gaze point) for frame index.

        Args:
            ind: Frame index

        Returns:
            Target point as float32 array
        """
        mainPt = self._dataset.storage[ind]["goal"]
        keypoints = np.array(mainPt, dtype=np.float32)
        return keypoints

    def _indexes2XY(
        self, indexesAndTime: List[Tuple[int, float]], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Dict, Dict], int]:
        """Convert frame indices and times to input/output tensors.

        Extracts data from storage and applies augmentation.

        Args:
            indexesAndTime: List of (frame_index, time) tuples
            kwargs: Parameters including timesteps and augmentation settings:
                - timesteps: Reshape timestep dimension
                - pointsNoise, pointsDropout: Face points augmentation
                - eyesAdditiveNoise, eyesDropout: Eye image augmentation
                - brightnessFactor, lightBlobFactor: Image augmentation
                - batch: Optional pre-allocated batch tuple (X, Y) from create_empty_batch()
                - batch_index: Index within batch to write data to (requires batch parameter)

        Returns:
            Tuple of ((X_dict, Y_dict), actual_size) where:
            - (X_dict, Y_dict): Data tuple where:
              - X_dict: Input data with clean/augmented variants and features
              - Y_dict: Target output points with 'result' key
            - actual_size: Number of sequences (B)

        Raises:
            AssertionError: If multiple user/place/screen IDs found
        """
        timesteps = kwargs.get("timesteps")
        samples = [self._dataset.storage[i] for i, _ in indexesAndTime]

        # Prepare target data
        y: np.ndarray = np.array(
            [self._targetFor(i) for i, _ in indexesAndTime], dtype=np.float32
        )
        y = self._reshapeSteps((y,), timesteps)[0]
        time: np.ndarray = np.array([T for _, T in indexesAndTime], dtype=np.float32)

        # NOTE: SamplesStorage guarantees ID consistency - all samples from same storage instance
        # have identical userId, screenId, cameraId, monitorId, placeId values. Extract once from first sample.
        user_id = samples[0]["userId"]
        screen_id = samples[0]["screenId"]
        camera_id = samples[0]["cameraId"]
        monitor_id = samples[0]["monitorId"]
        place_id = samples[0]["placeId"]

        # Prepare input data with augmentation
        X = DSUtils.toTensor(
            (
                np.array([x["points"] for x in samples], dtype=np.float32),
                np.array([x["left eye"] for x in samples]),
                np.array([x["right eye"] for x in samples]),
                time.reshape((-1, 1)),
            ),
            (
                kwargs.get("pointsNoise", 0.0),
                kwargs.get("pointsDropout", 0.0),
                kwargs.get("eyesAdditiveNoise", 0.0),
                kwargs.get("eyesDropout", 0.0),
                kwargs.get("brightnessFactor", 0.0),
                kwargs.get("lightBlobFactor", 0.0),
                kwargs.get("modalityDropout", 0.0),
                kwargs.get("regionFactor", 0.0),
                timesteps,
            ),
            user_id,
            screen_id,
            camera_id,
            monitor_id,
            place_id,
        )

        Y_tensor = tf.constant(value=y, dtype=tf.float32)

        # Return Y as dictionary with 'result' key
        Y = {"result": Y_tensor}

        # Calculate actual batch size
        B = len(y)

        # If batch pre-allocation provided, populate it and return the batch
        batch = kwargs.get("batch", None)
        batch_index = kwargs.get("batch_index", None)
        if batch is not None and batch_index is not None:
            batch_X, batch_Y = batch
            indices = [[batch_index + i] for i in range(B)]
            # Populate X tensors
            for k, v in X.items():
                if k in ("clean", "augmented"):
                    # For nested dicts (clean/augmented), populate each feature
                    for subk, subv in v.items():
                        batch_X[k][subk] = tf.tensor_scatter_nd_update(
                            batch_X[k][subk], indices, subv
                        )
                else:
                    # For flat dict keys
                    batch_X[k] = tf.tensor_scatter_nd_update(batch_X[k], indices, v)
            # Populate Y tensors
            for k, v in Y.items():
                batch_Y[k] = tf.tensor_scatter_nd_update(batch_Y[k], indices, v)
            return ((batch_X, batch_Y), B)

        return ((X, Y), B)

    def create_empty_batch(self, N: int, **kwargs: Any) -> Tuple[Dict, Dict]:
        """Create an empty batch with specified size and structure.

        Creates zero-filled TensorFlow tensors matching the expected output shape.

        Args:
            N: Batch size (number of sequences)
            **kwargs: Sampling parameters including:
                - timesteps: Number of frames per sequence

        Returns:
            Tuple of (X_dict, Y_dict) with zero-filled tensors of correct shape
        """
        kwargs = {**self._defaults, **kwargs}
        timesteps = kwargs.get("timesteps", None)
        assert timesteps is not None, "The number of timesteps must be defined."

        # Create empty X with clean/augmented structure
        X = {
            "clean": {
                "points": tf.zeros(
                    (N, timesteps, FACEMESH_LANDMARK_COUNT, 2), dtype=tf.float32
                ),
                "left eye": tf.zeros((N, timesteps, 32, 32, 1), dtype=tf.float32),
                "right eye": tf.zeros((N, timesteps, 32, 32, 1), dtype=tf.float32),
                "time": tf.zeros((N, timesteps, 1), dtype=tf.float32),
                "userId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "screenId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "cameraId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "monitorId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "placeId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "result": tf.zeros((N, timesteps, 2), dtype=tf.float32),
            },
            "augmented": {
                "points": tf.zeros(
                    (N, timesteps, FACEMESH_LANDMARK_COUNT, 2), dtype=tf.float32
                ),
                "left eye": tf.zeros((N, timesteps, 32, 32, 1), dtype=tf.float32),
                "right eye": tf.zeros((N, timesteps, 32, 32, 1), dtype=tf.float32),
                "time": tf.zeros((N, timesteps, 1), dtype=tf.float32),
                "userId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "screenId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "cameraId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "monitorId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "placeId": tf.zeros((N, timesteps, 1), dtype=tf.int32),
                "result": tf.zeros((N, timesteps, 2), dtype=tf.float32),
            },
        }

        # Create empty Y
        Y = {
            "result": tf.zeros((N, timesteps, 2), dtype=tf.float32),
        }

        return (X, Y)

    def merge(
        self, samples: List[Tuple[Dict, Dict]], expected_batch_size: int
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, tf.Tensor]]:
        """Merge multiple samples into a single batch.

        Concatenates X and Y from multiple samples along batch dimension using tf.concat,
        then shuffles the batch.

        Args:
            samples: List of (X, Y) tuples from sample() calls where Y is a dict
            expected_batch_size: Expected total batch size for validation

        Returns:
            Merged (X, Y) tuple with batch dimension (shuffled)

        Raises:
            AssertionError: If merged batch size doesn't match expected
        """
        # Convert all samples to tensors upfront
        tensor_samples = Utils.to_tensor(samples)

        # Shuffle batch indices to randomize order
        shuffle_indices = tf.random.shuffle(tf.range(expected_batch_size))

        # Merge Y as dictionary
        Y: Dict[str, tf.Tensor] = {}
        y_keys = samples[0][1].keys()

        for key in y_keys:
            # Extract converted tensors and concatenate
            y_tensors = [y[key] for _, y in tensor_samples]
            Y[key] = tf.concat(y_tensors, axis=0)
            Y[key] = tf.gather(Y[key], shuffle_indices)
            tf.debugging.assert_equal(
                tf.shape(Y[key])[0],
                expected_batch_size,
                message=f"Feature Y/{key} batch size mismatch: "
                f"got {tf.shape(Y[key])[0]}, expected {expected_batch_size}",
            )

        # X contains clean and augmented data
        # Format: {key: {subkey: tensor}} where key in ['clean', 'augmented']
        X: Dict[str, Dict[str, Any]] = {}
        feature_keys = samples[0][0]["clean"].keys()
        # Note: userId, placeId, screenId are included so ModelWrapper can compute embeddings

        for key in ["clean", "augmented"]:
            X[key] = {}
            for subkey in feature_keys:
                # Extract converted tensors and concatenate
                x_tensors = [x[key][subkey] for x, _ in tensor_samples]
                X[key][subkey] = tf.concat(x_tensors, axis=0)
                X[key][subkey] = tf.gather(X[key][subkey], shuffle_indices)

                tf.debugging.assert_equal(
                    tf.shape(X[key][subkey])[0],
                    expected_batch_size,
                    message=f"Feature {key}/{subkey} batch size mismatch: "
                    f"got {tf.shape(X[key][subkey])[0]}, expected {expected_batch_size}",
                )
        return (X, Y)
