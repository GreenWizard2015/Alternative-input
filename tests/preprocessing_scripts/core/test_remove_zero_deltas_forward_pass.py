"""Tests for remove_frames_with_zero_time_delta() forward pass (correctness of filtering)."""

import numpy as np
from scripts.preprocessing.core import remove_frames_with_zero_time_delta


class TestRemoveZeroDelatasForwardPass:
    """Test remove_frames_with_zero_time_delta() filtering correctness."""

    def test_no_duplicates_returns_unchanged(self):
        """Test that clean dataset with no duplicates remains unchanged."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert np.array_equal(
            result["time"], dataset["time"]
        ), "Clean data should remain unchanged"
        assert np.array_equal(
            result["data"], dataset["data"]
        ), "Data should match clean dataset"

    def test_removes_duplicate_timestamps(self):
        """Test removal of frames with duplicate timestamps.

        Times: [0.0, 0.1, 0.1, 0.2, 0.3]
        After filtering: [0.0, 0.1, 0.2, 0.3] (2nd 0.1 removed)
        """
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # One duplicate removed
        assert (
            len(result["time"]) == 4
        ), f"Expected 4 frames after removing duplicate, got {len(result['time'])}"
        assert np.all(
            np.diff(result["time"]) > 0.0
        ), "All deltas should be strictly positive"

    def test_removes_multiple_duplicates_iteratively(self):
        """Test iterative removal of multiple zero-delta frames.

        Times: [0.0, 0.1, 0.1, 0.2, 0.2]
        Iteration 1: Remove frame at index 2 (0.1==0.1) → [0.0, 0.1, 0.2, 0.2]
        Iteration 2: Remove frame at index 3 (0.2==0.2) → [0.0, 0.1, 0.2]
        Iteration 3: No more duplicates → stable
        """
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2, 0.2]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert (
            len(result["time"]) == 3
        ), f"Expected 3 frames after removing both duplicates, got {len(result['time'])}"
        assert np.array_equal(
            result["time"], np.array([0.0, 0.1, 0.2])
        ), "Should keep one of each duplicate"

    def test_preserves_data_ordering(self):
        """Test that data is filtered in same order as time."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),
            "data": np.array([10, 20, 30, 40]),  # Unique identifiers
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # Should keep frames at indices [0, 1, 3]
        # So data should be [10, 20, 40]
        expected_data = np.array([10, 20, 40])
        assert np.array_equal(
            result["data"], expected_data
        ), f"Expected {expected_data}, got {result['data']}"

    def test_multiple_fields_filtered_consistently(self):
        """Test that all fields are filtered to same indices."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4, 5]),
            "landmarks": np.array([100, 200, 300, 400, 500]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # All arrays should have same length
        time_len = len(result["time"])
        data_len = len(result["data"])
        landmarks_len = len(result["landmarks"])

        assert (
            time_len == data_len == landmarks_len
        ), f"Lengths mismatch: time={time_len}, data={data_len}, landmarks={landmarks_len}"

    def test_strict_comparison_not_approximate(self):
        """Test that comparison uses > 0.0, not approximate equality.

        Times with very small positive delta should be kept.
        Times: [0.0, 1e-15, 1e-10] should keep all (all deltas > 0)
        """
        dataset = {"time": np.array([0.0, 1e-15, 1e-10]), "data": np.array([1, 2, 3])}

        result = remove_frames_with_zero_time_delta(dataset)

        # All should be kept (deltas are technically positive)
        assert len(result["time"]) == 3, "Expected 3 time values"

    def test_convergence_count(self):
        """Test that function converges (doesn't loop indefinitely)."""
        # Create pathological case with many duplicates
        times = [0.0] + [0.1] * 100 + [0.2]
        dataset = {"time": np.array(times), "data": np.arange(len(times))}

        # Should converge and return valid result
        result = remove_frames_with_zero_time_delta(dataset)

        # Should end up with just [0.0, 0.1, 0.2]
        assert (
            len(result["time"]) == 3
        ), f"Should converge to 3 frames, got {len(result['time'])}"
        assert np.allclose(
            result["time"], [0.0, 0.1, 0.2]
        ), "Times should match expected values"

    def test_floating_point_precision(self):
        """Test handling of floating point precision edge cases.

        Tests that very small positive deltas are correctly identified as positive
        and frames are kept. Uses values that maintain strict monotonicity.
        """
        dataset = {
            "time": np.array(
                [0.0, 1e-15, 1e-10, 0.1]
            ),  # Very small but strictly increasing
            "data": np.array([1, 2, 3, 4]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # All should be kept (all deltas are strictly positive)
        assert len(result["time"]) == 4, "Expected 4 time values"
        assert np.all(
            np.diff(result["time"]) > 0.0
        ), "All deltas should be strictly positive"

    def test_large_dataset_performance(self):
        """Test with larger dataset to ensure reasonable performance."""
        # 10,000 frames with occasional duplicates
        times = np.sort(np.random.RandomState(42).uniform(0, 100, 10000))
        # Insert some exact duplicates
        times[100] = times[99]
        times[500] = times[499]
        times[1000] = times[999]

        dataset = {"time": times, "data": np.arange(len(times))}

        result = remove_frames_with_zero_time_delta(dataset)

        # Should remove duplicates
        assert len(result["time"]) < len(
            times
        ), "Result should have fewer frames after removing duplicates"
        assert np.all(
            np.diff(result["time"]) > 0.0
        ), "All remaining deltas should be positive"
