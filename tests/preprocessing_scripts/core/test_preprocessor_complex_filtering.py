"""Tests for complex filtering scenarios in process_dataset."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestProcessDatasetComplexFiltering:
    """Test process_dataset() with complex filtering scenarios."""

    def test_multiple_iterations_convergence(self):
        """Test that iterative zero-delta removal converges."""
        preprocessor = DatasetPreprocessor()

        # Create dataset with many duplicate patterns
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3]),
            "data": np.arange(7),
        }

        result = preprocessor.process_dataset(dataset)

        # Should converge to [0.0, 0.1, 0.2, 0.3]
        expected_times = np.array([0.0, 0.1, 0.2, 0.3])
        assert np.allclose(
            result["time"], expected_times
        ), "Time values should match expected after convergence"

    def test_non_array_fields_preserved(self):
        """Test that non-array fields pass through unchanged."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([1, 2, 3]),
            "metadata": "important_string",
            "count": 42,
        }

        result = preprocessor.process_dataset(dataset, validate_only=True)

        assert result["metadata"] == "important_string", "Metadata should be preserved"
        assert result["count"] == 42, "Count should be preserved"

    def test_different_temporal_scales(self):
        """Test with different temporal scales (seconds, milliseconds)."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=1.0)

        # Millisecond scale (0.001s = 1ms)
        dataset_ms = {
            "time": np.array([0.001, 0.002, 0.003, 0.004]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = preprocessor.process_dataset(dataset_ms, validate_only=True)
        assert len(result["time"]) == 4, "Expected 4 time values"

        # Large scale (1000s scale)
        dataset_large = {
            "time": np.array([0.0, 100.0, 200.0, 300.0]),
            "data": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(ValueError, match="too sparse"):
            preprocessor.process_dataset(dataset_large)

    def test_preserves_array_dtypes(self):
        """Test that array dtypes are preserved (except for filtering indices)."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "data": np.array([1, 2, 3], dtype=np.int32),
        }

        result = preprocessor.process_dataset(dataset, validate_only=True)

        # Dtypes should be preserved (though may change if filtered)
        assert (
            result["time"].dtype == np.float32
        ), "Time dtype should be preserved as float32"
        assert (
            result["data"].dtype == np.int32
        ), "Data dtype should be preserved as int32"

    def test_large_dataset_performance(self):
        """Test with larger dataset to verify reasonable performance."""
        preprocessor = DatasetPreprocessor()

        # 10,000 frames
        times = np.sort(np.random.RandomState(42).uniform(0, 100, 10000))
        dataset = {"time": times, "data": np.arange(len(times))}

        result = preprocessor.process_dataset(dataset, validate_only=True)

        assert len(result["time"]) > 0, "Result should have at least one frame"
        assert np.all(np.diff(result["time"]) >= 0), "Time should be non-decreasing"
