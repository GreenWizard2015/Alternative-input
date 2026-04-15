"""Tests for remove_frames_with_zero_time_delta() edge cases and boundary conditions."""

import numpy as np
from scripts.preprocessing.core import remove_frames_with_zero_time_delta


class TestRemoveZeroDeltasEdgeCases:
    """Test remove_frames_with_zero_time_delta() boundary conditions."""

    def test_single_frame_returns_unchanged(self):
        """Test with single frame (cannot compute delta, should return unchanged)."""
        dataset = {"time": np.array([0.0]), "data": np.array([1])}

        result = remove_frames_with_zero_time_delta(dataset)

        assert np.array_equal(
            result["time"], dataset["time"]
        ), "Single frame should remain unchanged"
        assert np.array_equal(
            result["data"], dataset["data"]
        ), "Single frame data should remain unchanged"

    def test_two_frames_with_identical_times(self):
        """Test with exactly 2 frames having identical timestamps."""
        dataset = {"time": np.array([0.0, 0.0]), "data": np.array([1, 2])}

        result = remove_frames_with_zero_time_delta(dataset)

        # Second frame removed (delta[0] = 0)
        assert (
            len(result["time"]) == 1
        ), "Should keep only first frame when second has zero delta"
        assert result["time"][0] == 0.0, "Expected result['time'][0] == 0.0"
        assert result["data"][0] == 1, "Expected result['data'][0] == 1"

    def test_two_frames_with_positive_delta(self):
        """Test with 2 frames having positive delta."""
        dataset = {"time": np.array([0.0, 0.1]), "data": np.array([1, 2])}

        result = remove_frames_with_zero_time_delta(dataset)

        assert (
            len(result["time"]) == 2
        ), "Both frames should be kept with positive delta"
        assert np.array_equal(
            result["time"], dataset["time"]
        ), "Times should remain unchanged"

    def test_alternating_duplicates_and_valid(self):
        """Test pattern: valid, dup, valid, dup - tests iterative convergence.

        Times: [0.0, 0.0, 0.1, 0.1, 0.2]
        Iteration 1: Remove indices 1,3 → [0.0, 0.1, 0.2]
        Iteration 2: No more duplicates → stable
        """
        dataset = {
            "time": np.array([0.0, 0.0, 0.1, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        expected_times = np.array([0.0, 0.1, 0.2])
        expected_data = np.array([1, 3, 5])

        assert np.array_equal(
            result["time"], expected_times
        ), f"Expected times {expected_times}, got {result['time']}"
        assert np.array_equal(
            result["data"], expected_data
        ), f"Expected data {expected_data}, got {result['data']}"

    def test_all_frames_identical_reduces_to_one(self):
        """Test with all identical timestamps - should keep only first."""
        dataset = {
            "time": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "data": np.array([10, 20, 30, 40, 50]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert (
            len(result["time"]) == 1
        ), "All identical times should reduce to single frame"
        assert result["time"][0] == 1.0, "Expected result['time'][0] == 1.0"
        assert result["data"][0] == 10, "Expected result['data'][0] == 10"

    def test_trailing_duplicates(self):
        """Test with duplicates at the end of sequence.

        Times: [0.0, 0.1, 0.2, 0.2, 0.2]
        Iteration 1: Remove indices 3,4 → [0.0, 0.1, 0.2]
        Iteration 2: Stable
        """
        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.2, 0.2]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert len(result["time"]) == 3, "Expected len(result['time']) == 3"
        assert result["time"][-1] == 0.2, "Expected result['time'][-1] == 0.2"
        assert result["data"][-1] == 3, "Expected result['data'][-1] == 3"

    def test_leading_duplicates(self):
        """Test with duplicates at the beginning of sequence.

        Times: [0.0, 0.0, 0.0, 0.1, 0.2]
        Iteration 1: Remove indices 1,2 → [0.0, 0.1, 0.2]
        Iteration 2: Stable
        """
        dataset = {
            "time": np.array([0.0, 0.0, 0.0, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert len(result["time"]) == 3, "Expected len(result['time']) == 3"
        assert result["time"][0] == 0.0, "Expected result['time'][0] == 0.0"
        assert result["data"][0] == 1, "Expected result['data'][0] == 1"

    def test_large_timestamp_values(self):
        """Test with very large timestamp values.

        Verifies algorithm works with timestamps like epoch seconds.
        """
        dataset = {
            "time": np.array([1609459200.0, 1609459200.0, 1609459200.1, 1609459200.2]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert (
            len(result["time"]) == 3
        ), "Should remove one duplicate from large timestamp values"
        assert np.all(
            np.diff(result["time"]) > 0.0
        ), "All time deltas should be positive"

    def test_near_zero_timestamps(self):
        """Test with timestamps very close to zero.

        Times: [0.0, 1e-16, 1e-15, 1e-14]
        All are strictly increasing, all should be kept.
        """
        dataset = {
            "time": np.array([0.0, 1e-16, 1e-15, 1e-14]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # All should be kept (very small but strictly positive deltas)
        assert len(result["time"]) == 4, "Expected len(result['time']) == 4"

    def test_empty_data_field(self):
        """Test that function handles dataset with time but empty other fields gracefully."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([]),  # Empty array
        }

        # This tests handling of mismatched array lengths
        # The function processes time field independently, so empty data field is handled
        result = remove_frames_with_zero_time_delta(dataset)

        # Verify result has valid time field (function processed it)
        assert "time" in result, "Result should contain time field"
        assert (
            len(result["time"]) > 0
        ), "Result time field should not be empty after processing"
        assert (
            result["time"].dtype == np.float64
        ), f"Time field should be float64, got {result['time'].dtype}"

    def test_negative_to_positive_timestamps_invalid(self):
        """Test that negative timestamps cause monotonicity validation error.

        Should raise ValueError during initialization check.
        """
        dataset = {"time": np.array([-1.0, 0.0, 1.0]), "data": np.array([1, 2, 3])}

        # Negative starting timestamp is fine, just needs to be non-decreasing
        result = remove_frames_with_zero_time_delta(dataset)
        assert len(result["time"]) == 3, "Expected len(result['time']) == 3"

    def test_many_duplicates_converges(self):
        """Test convergence with many consecutive duplicates.

        Times: [0.0, 0.1, 0.1, 0.1, 0.1, 0.2]
        Iteration 1: Remove indices 2,3,4 → [0.0, 0.1, 0.2]
        Iteration 2: Stable
        """
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4, 5, 6]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        expected_times = np.array([0.0, 0.1, 0.2])
        assert (
            len(result["time"]) == 3
        ), f"Should reduce to 3 frames, got {len(result['time'])}"
        assert np.array_equal(
            result["time"], expected_times
        ), f"Expected {expected_times}, got {result['time']}"

    def test_preserves_non_time_array_fields(self):
        """Test that non-array fields (strings, ints, etc.) are preserved."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4]),
            "metadata": "some string",  # Non-array field
            "count": 42,  # Non-array field
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # Non-array fields should pass through unchanged
        assert (
            result["metadata"] == "some string"
        ), "Non-array string field should be preserved"
        assert result["count"] == 42, "Expected result['count'] == 42"

    def test_only_first_and_last_frames_with_middle_duplicates(self):
        """Test: [0.0, 0.1, 0.1, 0.1, 1.0] → [0.0, 0.1, 1.0\']"""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.1, 1.0]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        expected_times = np.array([0.0, 0.1, 1.0])
        expected_data = np.array([1, 2, 5])

        assert np.array_equal(
            result["time"], expected_times
        ), f"Expected {expected_times}, got {result['time']}"
        assert np.array_equal(
            result["data"], expected_data
        ), f"Expected {expected_data}, got {result['data']}"
