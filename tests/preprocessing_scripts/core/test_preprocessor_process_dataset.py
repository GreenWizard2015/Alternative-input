"""Tests for DatasetPreprocessor.process_dataset() method."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestProcessDatasetBasic:
    """Test process_dataset() with various inputs."""

    def test_clean_dataset_passes_validation(self):
        """Test processing clean dataset (no zero deltas, good temporal density)."""
        preprocessor = DatasetPreprocessor(
            min_trajectory_frames=5, max_delta_threshold=0.3
        )

        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = preprocessor.process_dataset(dataset)

        assert len(result["time"]) == 5, "Expected 5 time values"
        assert np.array_equal(result["time"], dataset["time"])

    def test_dataset_with_duplicates_removed(self):
        """Test that zero-delta frames are removed."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = preprocessor.process_dataset(dataset)

        assert (
            len(result["time"]) == 4
        ), "Expected 4 time values after removing duplicates"
        assert np.all(
            np.diff(result["time"]) > 0
        ), "All time deltas should be positive (no duplicates)"

    def test_multiple_fields_processed_consistently(self):
        """Test that all fields are filtered consistently."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),
            "data": np.array([10, 20, 30, 40]),
            "landmarks": np.array([100, 200, 300, 400]),
        }

        result = preprocessor.process_dataset(dataset)

        assert len(result["time"]) == len(
            result["data"]
        ), "Fields should have same length"
        assert len(result["time"]) == len(
            result["landmarks"]
        ), "Time and landmarks should have same length"

    def test_validate_only_skips_filtering(self):
        """Test that validate_only=True doesn't modify data."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = preprocessor.process_dataset(dataset, validate_only=True)

        assert len(result["time"]) == 4, "Expected 4 time values"

    def test_sparse_dataset_fails_validation(self):
        """Test that overly sparse dataset fails validation."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=0.1)

        dataset = {
            "time": np.array([0.0, 0.5, 1.0, 1.5]),  # 0.5s gaps
            "data": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(ValueError, match="too sparse"):
            preprocessor.process_dataset(dataset)

    def test_missing_time_key_raises(self):
        """Test that missing 'time' key raises ValueError."""
        preprocessor = DatasetPreprocessor()

        dataset = {"data": np.array([1, 2, 3, 4]), "other": np.array([10, 20, 30, 40])}

        with pytest.raises(ValueError, match="time"):
            preprocessor.process_dataset(dataset)

    def test_mismatched_array_lengths_raise(self):
        """Test that mismatched array lengths raise ValueError."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([1, 2]),  # Different length
        }

        with pytest.raises(ValueError, match="length mismatch"):
            preprocessor.process_dataset(dataset)

    def test_empty_dataset_raises(self):
        """Test that empty time array raises error."""
        preprocessor = DatasetPreprocessor()

        dataset = {"time": np.array([]), "data": np.array([])}

        with pytest.raises(ValueError):
            preprocessor.process_dataset(dataset)

    def test_single_frame_dataset_fails_sparsity(self):
        """Test with single frame fails sparsity check (need >= 2 frames)."""
        preprocessor = DatasetPreprocessor()

        dataset = {"time": np.array([0.0]), "data": np.array([1])}

        with pytest.raises(ValueError, match="at least 2"):
            preprocessor.process_dataset(dataset, validate_only=True)
