"""Tests for remove_frames_with_zero_time_delta() initialization and parameter validation."""

import pytest
import numpy as np
from scripts.preprocessing.core import remove_frames_with_zero_time_delta


class TestRemoveZeroDelatasInitialization:
    """Test remove_frames_with_zero_time_delta() parameter validation."""

    def test_missing_time_key_raises(self):
        """Test that missing 'time' key raises ValueError."""
        dataset = {"data": np.array([1, 2, 3])}  # No 'time' key

        with pytest.raises(ValueError, match="time"):
            remove_frames_with_zero_time_delta(dataset)

    def test_empty_time_array_raises(self):
        """Test that empty 'time' array raises ValueError."""
        dataset = {"time": np.array([]), "data": np.array([])}

        with pytest.raises(ValueError, match="empty"):
            remove_frames_with_zero_time_delta(dataset)

    def test_single_frame_dataset_valid(self):
        """Test that single-frame dataset is accepted (no deltas to check)."""
        dataset = {"time": np.array([1.0]), "data": np.array([10])}

        result = remove_frames_with_zero_time_delta(dataset)

        assert result is not None, "Result should not be None for single frame"
        assert (
            len(result["time"]) == 1
        ), f"Single frame should remain, got {len(result['time'])} frames"

    def test_non_monotonic_times_raise(self):
        """Test that non-monotonic times raise ValueError."""
        dataset = {
            "time": np.array([0.0, 0.5, 0.3, 1.0]),  # Non-monotonic
            "data": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(ValueError, match="monotonic"):
            remove_frames_with_zero_time_delta(dataset)

    def test_strictly_increasing_times_valid(self):
        """Test that strictly increasing times are accepted."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert result is not None, "Result should not be None for valid input"
        assert (
            len(result["time"]) == 4
        ), f"All 4 frames should remain, got {len(result['time'])}"

    def test_non_decreasing_times_valid(self):
        """Test that non-strictly-increasing (non-decreasing) times are accepted initially.

        Duplicates should be handled during filtering, not validation.
        """
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),  # Non-strictly increasing
            "data": np.array([1, 2, 3, 4]),
        }

        # Should not raise during validation - duplicates handled in filtering
        result = remove_frames_with_zero_time_delta(dataset)
        assert result is not None, "Result should not be None for non-decreasing times"

    def test_preserves_data_structure(self):
        """Test that function preserves dict structure and all keys."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([1, 2, 3]),
            "landmarks": np.array([10, 20, 30]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        assert "time" in result, "Result should contain 'time' key"
        assert "data" in result, "Result should contain 'data' key"
        assert "landmarks" in result, "Result should contain 'landmarks' key"

    def test_does_not_modify_input(self):
        """Test that input dataset is not modified."""
        original_data = np.array([0.0, 0.1, 0.2])
        dataset = {"time": original_data.copy(), "data": np.array([1, 2, 3])}

        _ = remove_frames_with_zero_time_delta(dataset)

        # Original should be unchanged
        assert np.array_equal(
            dataset["time"], original_data
        ), "Input dataset should not be modified"

    def test_multiple_additional_fields(self):
        """Test with many additional fields beyond 'time'."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4]),
            "field1": np.array([10, 20, 30, 40]),
            "field2": np.array([100, 200, 300, 400]),
            "field3": np.array([1000, 2000, 3000, 4000]),
        }

        result = remove_frames_with_zero_time_delta(dataset)

        # All fields should be present
        assert len(result) == 5, f"Should have 5 fields, got {len(result)}"
        # All fields should have same length
        expected_len = len(result["time"])
        for key in result:
            assert (
                len(result[key]) == expected_len
            ), f"Field '{key}' should have {expected_len} elements, got {len(result[key])}"
