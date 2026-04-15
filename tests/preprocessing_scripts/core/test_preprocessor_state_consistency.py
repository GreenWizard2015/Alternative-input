"""Tests for DatasetPreprocessor state consistency."""

import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestDatasetPreprocessorStateConsistency:
    """Test that DatasetPreprocessor maintains state consistency."""

    def test_parameters_unchanged_after_operations(self):
        """Test that parameters remain unchanged after preprocessing."""
        preprocessor = DatasetPreprocessor(
            min_trajectory_frames=5, max_delta_threshold=0.3
        )

        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4]),
        }

        # Process dataset (ignore result since we're testing parameter preservation)
        preprocessor.process_dataset(dataset)

        # Parameters should be unchanged
        assert (
            preprocessor.min_trajectory_frames == 5
        ), "min_trajectory_frames should remain 5"
        assert (
            preprocessor.max_delta_threshold == 0.3
        ), "max_delta_threshold should remain 0.3"

    def test_same_preprocessor_multiple_operations(self):
        """Test that same preprocessor can be used for multiple datasets."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        dataset1 = {
            "time": np.array([0.0, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4]),
        }
        dataset2 = {
            "time": np.array([0.0, 0.05, 0.1, 0.15]),
            "data": np.array([10, 20, 30, 40]),
        }

        # Both should process successfully with same config
        result1 = preprocessor.process_dataset(dataset1, validate_only=True)
        result2 = preprocessor.process_dataset(dataset2, validate_only=True)

        assert (
            "time" in result1 and "time" in result2
        ), "Both results should contain 'time' key"
