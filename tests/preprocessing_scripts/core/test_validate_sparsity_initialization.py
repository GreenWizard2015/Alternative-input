"""Tests for validate_dataset_sparsity() initialization and parameter validation."""

import pytest
import numpy as np
from scripts.preprocessing.core import validate_dataset_sparsity


class TestValidateDatasetSparsityInitialization:
    """Test validate_dataset_sparsity() parameter validation."""

    def test_valid_initialization_with_standard_params(self):
        """Test successful validation with valid parameters."""
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        # Should not raise - min delta (0.1) < threshold (0.3)
        validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_returns_none_on_success(self):
        """Test that function returns None when validation passes."""
        times = np.array([0.0, 0.1, 0.2])

        result = validate_dataset_sparsity(times, max_delta_threshold=0.3)

        assert result is None, "Function should return None on successful validation"

    def test_single_frame_raises_error(self):
        """Test that single frame (< 2 frames) raises ValueError."""
        times = np.array([0.0])

        with pytest.raises(ValueError, match="at least 2 frames"):
            validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        times = np.array([])

        with pytest.raises(ValueError, match="at least 2 frames"):
            validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_negative_threshold_value(self):
        """Test with negative threshold - passes validation check (no frames too sparse)."""
        times = np.array([0.0, 0.1, 0.2])

        # Negative threshold means no frame spacing can be too sparse
        # (since min_delta 0.1 > -0.1 is true)
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=-0.1)

    def test_zero_threshold_value(self):
        """Test with zero threshold - any positive delta exceeds it."""
        times = np.array([0.0, 0.1, 0.2])

        # min_delta (0.1) > threshold (0.0), should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.0)

    def test_very_large_threshold(self):
        """Test with very large threshold - should always pass."""
        times = np.array([0.0, 100.0, 200.0, 300.0])  # Large gaps

        # min_delta (100.0) < threshold (1e10), should pass
        validate_dataset_sparsity(times, max_delta_threshold=1e10)

    def test_threshold_exactly_equals_min_delta(self):
        """Test when threshold exactly equals minimum delta.

        When threshold == min_delta, the condition is min_delta > threshold
        which is False, so validation passes.
        """
        times = np.array([0.0, 0.1, 0.2, 0.3])
        # min_delta (0.1) > threshold (0.1) is False, so should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_threshold_slightly_less_than_min_delta(self):
        """Test when threshold is slightly less than minimum delta."""
        times = np.array([0.0, 0.11, 0.22, 0.33])
        # min_delta (0.11) > threshold (0.1), should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_threshold_slightly_more_than_min_delta(self):
        """Test when threshold is slightly more than minimum delta."""
        times = np.array([0.0, 0.1, 0.2, 0.3])

        # min_delta (0.1) > threshold (0.11) is False, should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.11)

    def test_two_frame_dataset_valid(self):
        """Test with exactly 2 frames (minimum valid case)."""
        times = np.array([0.0, 0.1])

        # delta = 0.1, threshold = 0.3
        validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_large_dataset_with_small_deltas(self):
        """Test with large dataset having small, uniform deltas."""
        times = np.linspace(0.0, 10.0, 1000)  # 1000 frames, 0.01 delta each

        # min_delta ≈ 0.01 < threshold (0.3), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_preserves_input_array(self):
        """Test that validation doesn't modify input array."""
        times = np.array([0.0, 0.1, 0.2, 0.3])
        times_original = times.copy()

        validate_dataset_sparsity(times, max_delta_threshold=0.3)

        assert np.array_equal(
            times, times_original
        ), "Input array should not be modified"

    def test_non_uniform_deltas_uses_minimum(self):
        """Test that validation uses minimum delta, not average.

        Times: [0.0, 0.1, 1.0, 10.0]
        Deltas: [0.1, 0.9, 9.0]
        Min delta: 0.1
        """
        times = np.array([0.0, 0.1, 1.0, 10.0])

        # min_delta (0.1) < threshold (0.3), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.3)

        # But min_delta (0.1) > threshold (0.05), should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.05)

    def test_floating_point_times(self):
        """Test with floating point timestamp values."""
        times = np.array([0.123456, 0.234567, 0.345678, 0.456789])

        # All deltas should be approximately 0.111
        validate_dataset_sparsity(times, max_delta_threshold=0.12)

    def test_large_epoch_timestamps(self):
        """Test with epoch (unix timestamp) values."""
        times = np.array(
            [
                1609459200.0,  # 2021-01-01 00:00:00
                1609459260.0,  # +60 seconds
                1609459320.0,  # +60 seconds
            ]
        )

        # min_delta = 60 seconds
        validate_dataset_sparsity(times, max_delta_threshold=100.0)
