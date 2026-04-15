"""Tests for get_trajectory_context method."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestGetTrajectoryContext:
    """Test get_trajectory_context() method."""

    def test_extract_trajectory_context(self):
        """Test extracting trajectory context for a sample."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "data": np.array([1, 2, 3, 4, 5, 6, 7]),
        }

        context = preprocessor.get_trajectory_context(dataset, sample_index=5)

        # Sample 5 needs frames [1, 2, 3, 4, 5]
        expected_data = np.array([2, 3, 4, 5, 6])
        assert np.array_equal(
            context["data"], expected_data
        ), "Context data should match expected trajectory"

    def test_context_has_correct_length(self):
        """Test that context has correct frame count."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=7)

        dataset = {"time": np.linspace(0, 1, 20), "data": np.arange(20)}

        context = preprocessor.get_trajectory_context(dataset, sample_index=10)

        assert len(context["time"]) == 7, "Context should have 7 time frames"
        assert len(context["data"]) == 7, "Context should have 7 data frames"

    def test_context_includes_sample_frame(self):
        """Test that context includes the sample frame itself."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        dataset = {"time": np.arange(10, dtype=float), "data": np.arange(10) * 10}

        context = preprocessor.get_trajectory_context(dataset, sample_index=7)

        # Context should end at sample_index
        assert context["data"][-1] == 70, "Context should end with sample data"

    def test_sample_too_early_raises(self):
        """Test that sample too close to start raises."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        dataset = {"time": np.arange(10, dtype=float), "data": np.arange(10)}

        with pytest.raises(ValueError, match="trajectory|require"):
            preprocessor.get_trajectory_context(dataset, sample_index=2)

    def test_context_with_multiple_fields(self):
        """Test context extraction with multiple fields."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=3)

        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "data": np.array([1, 2, 3, 4, 5]),
            "landmarks": np.array([10, 20, 30, 40, 50]),
        }

        context = preprocessor.get_trajectory_context(dataset, sample_index=3)

        assert len(context) == 3, "Context should have 3 fields"
        assert len(context["data"]) == 3, "Context data should have 3 elements"
        assert (
            len(context["landmarks"]) == 3
        ), "Context landmarks should have 3 elements"
        assert np.array_equal(
            context["data"], np.array([2, 3, 4])
        ), "Context data should match expected values"
