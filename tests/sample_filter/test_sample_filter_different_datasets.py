"""Tests for SampleFilter with different datasets in different instances."""

from Core.data.SampleFilter import SampleFilter
from tests.fixtures.sample_filter_fixtures import _MockStorage


class TestSampleFilterDifferentDatasets:
    """Test behavior with different datasets in different instances."""

    def test_different_instances_different_datasets_different_results(self):
        """Test that different instances with different datasets return different results.

        This is important: if instance1 and instance2 have DIFFERENT storage,
        they SHOULD return different results for the same index.
        """
        # Dataset A: 10 frames at 0.1s intervals
        storage_a = _MockStorage([i * 0.1 for i in range(10)])
        filter_a = SampleFilter(storage_a, minFrames=1, maxT=1.0)

        # Dataset B: only 3 frames at 1.0s intervals
        storage_b = _MockStorage([0.0, 1.0, 2.0])
        filter_b = SampleFilter(storage_b, minFrames=1, maxT=1.0)

        # Both call trajectory_start(2) on their respective datasets
        result_a = filter_a.trajectory_start(2)  # From 10-frame dataset
        result_b = filter_b.trajectory_start(2)  # From 3-frame dataset

        # Results SHOULD be different because datasets are different
        assert (
            result_a != result_b
        ), f"Different datasets should give different results for trajectory_start(2): Dataset A (dense)={result_a}, Dataset B (sparse)={result_b}"
        # Dataset A: frame 2 at time=0.2, window ±1.0s includes all frames 0-9, starts at 0
        assert (
            result_a == 0
        ), f"Dataset A result should be 0 (frame 2 at 0.2s with maxT=1.0 includes frames 0-9), got {result_a}"
        # Dataset B: frame 2 at time=2.0, window ±1.0s includes frames with time 1.0-3.0
        # That's frames 1,2 (times 1.0, 2.0). Frame 0 (time=0.0) is outside window, starts at 1
        assert (
            result_b == 1
        ), f"Dataset B result should be 1 (frame 2 at 2.0s with maxT=1.0 includes frames 1-2), got {result_b}"

    def test_same_instance_same_index_same_result(self):
        """Test that same instance returns same result for same index (basic cache test)."""
        storage = _MockStorage([i * 0.1 for i in range(10)])
        filter_obj = SampleFilter(storage, minFrames=1, maxT=1.0)

        result1 = filter_obj.trajectory_start(5)
        result2 = filter_obj.trajectory_start(5)  # Should hit cache

        assert (
            result1 == result2
        ), f"Same instance should return same result for trajectory_start(5) due to caching, got {result1} vs {result2}"
