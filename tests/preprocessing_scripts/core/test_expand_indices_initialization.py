"""Tests for expand_indices_for_trajectories() initialization and parameter validation."""

import pytest
import numpy as np
from scripts.preprocessing.core import expand_indices_for_trajectories


class TestExpandIndicesInitialization:
    """Test expand_indices_for_trajectories() parameter validation."""

    def test_valid_initialization_with_standard_params(self):
        """Test successful call with valid parameters."""
        indices = np.array([10, 20, 30])
        result = expand_indices_for_trajectories(indices, 5)

        assert result is not None, "Function should return non-None result"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert len(result) > 0, "Result should have elements"

    def test_negative_min_frames_raises(self):
        """Test that negative min_frames raises ValueError."""
        indices = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="positive"):
            expand_indices_for_trajectories(indices, -1)

    def test_zero_min_frames_raises(self):
        """Test that min_frames=0 raises ValueError."""
        indices = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="positive"):
            expand_indices_for_trajectories(indices, 0)

    def test_single_sample_index(self):
        """Test with single sample (array of length 1)."""
        indices = np.array([10])
        result = expand_indices_for_trajectories(indices, 5)

        assert len(result) > 0, f"Single sample should produce output, got {result}"
        assert 10 in result, f"Sample index should be in result, got {result}"

    def test_empty_indices_array(self):
        """Test with empty indices array."""
        indices = np.array([], dtype=np.int64)
        result = expand_indices_for_trajectories(indices, 5)

        assert len(result) == 0, "Empty input should produce empty output"

    def test_min_frames_equals_1(self):
        """Test with min_frames=1 (minimal trajectory)."""
        indices = np.array([10, 20, 30])
        result = expand_indices_for_trajectories(indices, 1)

        # With min_frames=1, each sample needs 1 frame (itself only)
        assert np.all(
            np.isin(indices, result)
        ), f"All samples should be in result, missing: {indices[~np.isin(indices, result)]}"
        assert len(result) >= len(
            indices
        ), f"Result should have at least as many elements, got {len(result)} from {len(indices)} samples"

    def test_large_min_frames(self):
        """Test with large min_frames value."""
        indices = np.array([100])
        result = expand_indices_for_trajectories(indices, 50)

        # Should compute frames [100 - 50 + 1, ..., 100] = [51, ..., 100]
        assert (
            len(result) >= 50
        ), f"Should include all trajectory frames, got {len(result)}: {result}"
        assert 100 in result, f"Target sample should be in result, got {result}"
        assert 51 in result, f"Start of trajectory should be in result, got {result}"
        assert (
            50 not in result
        ), f"Frame before trajectory should not be included, got {result}"

    def test_output_is_not_reference(self):
        """Test that returned array is independent copy, not reference."""
        indices = np.array([10, 20, 30])
        result = expand_indices_for_trajectories(indices, 5)

        # Verify it's a new array (not mutated version of input)
        assert result is not indices, "Result should not be same object as input"
        # Modify result
        result[0] = -999
        # Input should be unchanged
        assert indices[0] != -999, "Modifying result should not affect input"
