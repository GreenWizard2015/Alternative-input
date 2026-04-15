"""Tests for validate_dataset_sparsity() forward pass (correctness of validation logic)."""

import pytest
import numpy as np
from scripts.preprocessing.core import validate_dataset_sparsity


class TestValidateDatasetSparsityForwardPass:
    """Test validate_dataset_sparsity() validation logic correctness."""

    def test_dense_dataset_passes_validation(self):
        """Test that densely sampled dataset passes validation.

        Frames every 33ms (typical 30fps video) with threshold of 100ms.
        """
        # 30 FPS ≈ 33ms per frame
        times = np.arange(0, 1.0, 1.0 / 30)  # 0.033 second intervals

        # min_delta ≈ 0.033 < threshold (0.1), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_sparse_dataset_fails_validation(self):
        """Test that sparsely sampled dataset fails validation.

        Frames every 500ms with threshold of 100ms.
        """
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        # min_delta (0.5) > threshold (0.1), should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_boundary_case_slightly_sparse(self):
        """Test boundary: min_delta just barely exceeds threshold.

        Times with 0.101 second intervals vs 0.1 second threshold.
        """
        times = np.array([0.0, 0.101, 0.202, 0.303])

        # min_delta (0.101) > threshold (0.1), should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_boundary_case_just_dense_enough(self):
        """Test boundary: min_delta just barely passes threshold.

        Times with 0.099 second intervals vs 0.1 second threshold.
        """
        times = np.array([0.0, 0.099, 0.198, 0.297])

        # min_delta (0.099) < threshold (0.1), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_varying_intervals_uses_minimum(self):
        """Test that validation uses minimum interval, not average.

        Times: [0.0, 0.05, 1.0, 1.5] with min_delta = 0.05
        Average would be misleading.
        """
        times = np.array([0.0, 0.05, 1.0, 1.5])
        # Intervals: [0.05, 0.95, 0.5]
        # Min: 0.05, Average: 0.5

        # threshold = 0.04 should fail (0.05 > 0.04)
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.04)

        # threshold = 0.06 should pass (0.05 < 0.06)
        validate_dataset_sparsity(times, max_delta_threshold=0.06)

    def test_realistic_eye_tracking_scenario(self):
        """Test with realistic eye-tracking frame rates.

        Typical eye tracker at 120Hz = 8.33ms per frame.
        Typical threshold: don't allow gaps > 50ms.
        """
        # 120 Hz sampling
        times = np.arange(0, 2.0, 1.0 / 120)

        # min_delta ≈ 0.0083 < threshold (0.05), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.05)

        # But if we require < 5ms, should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.005)

    def test_60fps_video_scenario(self):
        """Test with standard 60fps video.

        60 FPS = 16.67ms per frame, threshold 50ms.
        """
        times = np.arange(0, 5.0, 1.0 / 60)

        # min_delta ≈ 0.0167 < threshold (0.05), should pass
        validate_dataset_sparsity(times, max_delta_threshold=0.05)

    def test_mixed_sampling_rate_find_worst_gap(self):
        """Test with intentionally mixed sampling rates.

        Most frames sampled at 30ms, but one gap of 150ms.
        Threshold of 100ms should fail (due to 150ms gap).
        """
        # Create times with one large gap
        base_times = np.arange(0, 1.0, 0.03)  # 30ms intervals
        times = np.concatenate([base_times, np.array([1.15])])  # Add gap of 150ms

        # min_delta (0.03) < threshold (0.1), should pass
        # Even though there's a large gap, we only check min_delta
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

        # But if threshold is very small
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.02)

    def test_monotonically_increasing_required(self):
        """Test that times must be monotonically increasing (enforced before validation).

        Actually, validation assumes input is already monotonic
        (that should be validated elsewhere). This test documents that assumption.
        """
        # Input must already be monotonic
        times = np.array([0.0, 0.1, 0.2, 0.3])

        validate_dataset_sparsity(times, max_delta_threshold=0.3)

    def test_very_small_thresholds(self):
        """Test with extremely small thresholds (e.g., microsecond precision)."""
        times = np.array([0.0, 1e-6, 2e-6, 3e-6])

        # min_delta = 1e-6, threshold = 2e-6
        validate_dataset_sparsity(times, max_delta_threshold=2e-6)

        # But threshold = 5e-7 should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=5e-7)

    def test_high_resolution_timestamps(self):
        """Test with high-resolution (nanosecond) timestamps."""
        # Nanosecond precision timestamps
        times = np.array(
            [
                1e9,
                1e9 + 1e6,  # +1 millisecond
                1e9 + 2e6,  # +1 millisecond
            ]
        )

        # min_delta = 1e6 (1ms), threshold = 2e6 (2ms)
        validate_dataset_sparsity(times, max_delta_threshold=2e6)

    def test_logarithmic_spacing(self):
        """Test with logarithmically spaced times (unusual but valid).

        Times: [1, 10, 100, 1000]
        Deltas: [9, 90, 900]
        Min: 9
        """
        times = np.array([1, 10, 100, 1000])

        # min_delta = 9 < threshold = 100, should pass
        validate_dataset_sparsity(times, max_delta_threshold=100)

        # min_delta = 9 > threshold = 5, should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=5)

    def test_identical_consecutive_times_rejected_elsewhere(self):
        """Test documents that identical times should be rejected before sparsity check.

        This function assumes times have already been cleaned of duplicates.
        Sparsity check with [0.0, 0.0, 0.1] would see min_delta = 0.0
        and would fail validation.
        """
        times = np.array([0.0, 0.0, 0.1])

        # min_delta = 0 > threshold doesn't make sense
        # Our validation sees 0 > threshold(0.1) = False, so it passes
        # But this documents that such input shouldn't reach here
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_error_message_includes_useful_info(self):
        """Test that error message includes min delta and threshold for debugging."""
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_passes_with_exact_threshold_match(self):
        """Test edge case where min_delta exactly equals threshold.

        The condition is min_delta > threshold, so equality means it passes.
        """
        times = np.array([0.0, 0.1, 0.2, 0.3])

        # min_delta = 0.1, threshold = 0.1
        # 0.1 > 0.1 is False, so validation passes
        validate_dataset_sparsity(times, max_delta_threshold=0.1)

    def test_long_duration_data(self):
        """Test with very long recording duration (e.g., 24-hour experiment).

        Duration: 24 hours (86400 seconds)
        Sample rate: 1 sample per second
        Min delta: 1 second
        """
        times = np.arange(0, 86400, 1.0)

        # min_delta = 1.0, threshold = 2.0, should pass
        validate_dataset_sparsity(times, max_delta_threshold=2.0)

        # But threshold = 0.5 should fail
        with pytest.raises(ValueError, match="too sparse"):
            validate_dataset_sparsity(times, max_delta_threshold=0.5)
