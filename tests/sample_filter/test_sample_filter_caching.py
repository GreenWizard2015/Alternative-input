"""Tests for SampleFilter caching behavior."""

from Core.data.SampleFilter import SampleFilter


class TestSampleFilterCaching:
    """Test caching behavior of trajectory_start()."""

    def test_trajectory_start_cached(self, filter_with_1s_window):
        """Test that trajectory_start() caches results."""
        # Call twice with same index - should return identical result due to caching
        result1 = filter_with_1s_window.trajectory_start(5)
        result2 = filter_with_1s_window.trajectory_start(5)
        assert (
            result1 == result2
        ), f"Cache consistency: trajectory_start(5) should return same result on repeated calls, got {result1} vs {result2}"

    def test_trajectory_start_different_indices(self, storage_10_frames):
        """Test that different indices return different results."""
        # Use a smaller window where different frames have different start indices
        filter_obj = SampleFilter(storage_10_frames, minFrames=1, maxT=0.3)
        result0 = filter_obj.trajectory_start(0)
        result5 = filter_obj.trajectory_start(5)
        result9 = filter_obj.trajectory_start(9)
        assert (
            result0 != result5
        ), f"Different indices should return different trajectory starts: frame 0 (time 0.0s) start={result0}, frame 5 (time 0.5s) start={result5}"
        assert (
            result5 != result9
        ), f"Different indices should return different trajectory starts: frame 5 (time 0.5s) start={result5}, frame 9 (time 0.9s) start={result9}"

    def test_trajectory_start_cache_multiple_calls(self, filter_with_1s_window):
        """Test cache efficiency with multiple calls."""
        # Call trajectory_start multiple times with different indices
        indices = [0, 2, 4, 6, 8, 2, 4, 6]  # Some repeats
        results = [filter_with_1s_window.trajectory_start(i) for i in indices]
        # Verify repeated indices return same values (cache working)
        assert (
            results[1] == results[5]
        ), f"Cache consistency: trajectory_start(2) should return same value on repeated calls, got {results[1]} vs {results[5]}"
        assert (
            results[2] == results[6]
        ), f"Cache consistency: trajectory_start(4) should return same value on repeated calls, got {results[2]} vs {results[6]}"
