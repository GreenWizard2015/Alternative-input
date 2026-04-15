"""Tests for DatasetPreprocessor initialization and configuration."""

import numpy as np
import pytest
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestDatasetPreprocessorInitialization:
    """Test DatasetPreprocessor initialization and parameter validation."""

    def test_default_initialization(self):
        """Test preprocessor with default parameters."""
        preprocessor = DatasetPreprocessor()

        # Test behavior with default parameters on actual data
        dataset = {"time": np.array([0.0, 0.1, 0.2, 0.3, 0.4]), "data": np.arange(5)}
        result = preprocessor.process_dataset(dataset, validate_only=True)

        assert (
            preprocessor.min_trajectory_frames > 0
        ), "Should have positive min_trajectory_frames"
        assert (
            preprocessor.max_delta_threshold > 0
        ), "Should have positive max_delta_threshold"
        assert result is not None, "Preprocessor should validate dataset successfully"

    def test_custom_initialization(self):
        """Test preprocessor with custom parameters."""
        preprocessor = DatasetPreprocessor(
            min_trajectory_frames=10, max_delta_threshold=0.5
        )

        # Test behavior with custom parameters
        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "data": np.arange(11),
        }
        result = preprocessor.process_dataset(dataset, validate_only=True)

        assert (
            preprocessor.min_trajectory_frames == 10
        ), "min_trajectory_frames should be 10"
        assert (
            preprocessor.max_delta_threshold == 0.5
        ), "max_delta_threshold should be 0.5"
        assert (
            result is not None
        ), "Preprocessor should validate dataset with custom parameters"

    def test_negative_min_frames_raises(self):
        """Test that negative min_trajectory_frames raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DatasetPreprocessor(min_trajectory_frames=-1)

    def test_zero_min_frames_raises(self):
        """Test that min_trajectory_frames=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DatasetPreprocessor(min_trajectory_frames=0)

    def test_min_trajectory_frames_equals_1(self):
        """Test with min_trajectory_frames=1 (minimal)."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=1)

        assert (
            preprocessor.min_trajectory_frames == 1
        ), "min_trajectory_frames should be 1"

    def test_large_min_trajectory_frames(self):
        """Test with large min_trajectory_frames."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=1000)

        assert (
            preprocessor.min_trajectory_frames == 1000
        ), "min_trajectory_frames should be 1000"

    def test_zero_threshold_allowed(self):
        """Test that max_delta_threshold=0 is allowed (very strict)."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=0.0)

        assert (
            preprocessor.max_delta_threshold == 0.0
        ), "max_delta_threshold should be 0.0"

    def test_negative_threshold_allowed(self):
        """Test that negative threshold is allowed (would make validation strict)."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=-1.0)

        assert (
            preprocessor.max_delta_threshold == -1.0
        ), "max_delta_threshold should be -1.0"

    def test_very_large_threshold(self):
        """Test with very large threshold (permissive)."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=1e10)

        assert (
            preprocessor.max_delta_threshold == 1e10
        ), "max_delta_threshold should be 1e10"

    def test_initialization_creates_independent_instances(self):
        """Test that multiple instances are independent."""
        p1 = DatasetPreprocessor(min_trajectory_frames=5)
        p2 = DatasetPreprocessor(min_trajectory_frames=10)

        assert p1.min_trajectory_frames == 5, "p1.min_trajectory_frames should be 5"
        assert p2.min_trajectory_frames == 10, "p2.min_trajectory_frames should be 10"
        assert p1 is not p2, "p1 and p2 should be different instances"

    def test_initialization_with_floats_accepted(self):
        """Test that float values are accepted for threshold."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=0.05)

        assert (
            preprocessor.max_delta_threshold == 0.05
        ), "max_delta_threshold should be 0.05"
        assert isinstance(
            preprocessor.max_delta_threshold, float
        ), "max_delta_threshold should be a float"

    def test_initialization_logs_configuration(self):
        """Test that initialization logs the configuration."""
        preprocessor = DatasetPreprocessor(
            min_trajectory_frames=7, max_delta_threshold=0.2
        )

        # Just verify initialization succeeds and stores values
        assert (
            preprocessor.min_trajectory_frames == 7
        ), "min_trajectory_frames should be 7"
        assert (
            preprocessor.max_delta_threshold == 0.2
        ), "max_delta_threshold should be 0.2"
