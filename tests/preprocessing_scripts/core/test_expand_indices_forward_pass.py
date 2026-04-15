"""Tests for expand_indices_for_trajectories() forward pass (correctness of computation)."""

import numpy as np
from scripts.preprocessing.core import expand_indices_for_trajectories


class TestExpandIndicesForwardPass:
    """Test that expand_indices_for_trajectories() computes correct frame indices."""

    def test_simple_expansion(self):
        """Test basic expansion with single sample.

        Sample at index 10, min_frames=5 should give frames [6,7,8,9,10]
        """
        indices = np.array([10])
        result = expand_indices_for_trajectories(indices, 5)

        expected = np.array([6, 7, 8, 9, 10])
        assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    def test_multiple_non_overlapping_samples(self):
        """Test multiple samples with non-overlapping trajectories.

        Samples: [10, 20]
        Min frames: 5
        Each needs 5 frames before it: [6-10] and [16-20]
        """
        indices = np.array([10, 20])
        result = expand_indices_for_trajectories(indices, 5)

        # Should include both trajectory windows
        assert 6 in result, "Start of first trajectory should be present"
        assert 10 in result, "First sample should be present"
        assert 16 in result, "Start of second trajectory should be present"
        assert 20 in result, "Second sample should be present"

        # No overlap between [6-10] and [16-20]
        assert 11 not in result, "Gap between trajectories should have no frames"
        assert 15 not in result, "Gap between trajectories should have no frames"

    def test_overlapping_trajectories(self):
        """Test samples close enough to have overlapping trajectories.

        Samples: [10, 12]
        Min frames: 5
        Trajectories: [6-10] and [8-12]
        Union: [6,7,8,9,10,11,12]
        """
        indices = np.array([10, 12])
        result = expand_indices_for_trajectories(indices, 5)

        expected = np.array([6, 7, 8, 9, 10, 11, 12])
        assert np.array_equal(
            result, expected
        ), f"Overlapping trajectories: expected {expected}, got {result}"

    def test_min_frames_2(self):
        """Test with min_frames=2 (shorter trajectories).

        Sample at 10 with min_frames=2 needs frames [9,10]
        """
        indices = np.array([10])
        result = expand_indices_for_trajectories(indices, 2)

        expected = np.array([9, 10])
        assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    def test_result_is_sorted_ascending(self):
        """Test that output is sorted in ascending order.

        This is a key CONTRACT - expand_indices() must always return sorted output.
        """
        indices = np.array([50, 10, 30, 20])  # Unsorted input
        result = expand_indices_for_trajectories(indices, 5)

        # Verify sorted
        assert np.all(
            result[:-1] < result[1:]
        ), f"Result should be strictly ascending, got {result}"

        # Verify no duplicates (consequence of np.unique())
        assert len(result) == len(
            np.unique(result)
        ), f"Result should have no duplicates, got {result}"

    def test_result_has_no_duplicates(self):
        """Test that result has no duplicate indices."""
        indices = np.array([10, 11, 12])  # Consecutive samples
        result = expand_indices_for_trajectories(indices, 5)

        # np.unique() should eliminate duplicates
        unique_result = np.unique(result)
        assert len(result) == len(
            unique_result
        ), f"Result should have no duplicates: {result}, unique: {unique_result}"

    def test_all_samples_in_output(self):
        """Test that all input samples appear in output.

        Every sample index should be in the expanded set (as the end of its trajectory).
        """
        indices = np.array([10, 50, 100])
        result = expand_indices_for_trajectories(indices, 5)

        for idx in indices:
            assert idx in result, f"Sample index {idx} should be in output"

    def test_large_sample_count(self):
        """Test with many samples."""
        indices = np.arange(100, 1000, 50)  # [100, 150, 200, ..., 950]
        result = expand_indices_for_trajectories(indices, 10)

        # Verify structure
        assert len(result) > len(
            indices
        ), f"Expansion should increase count, got {len(result)} from {len(indices)} samples"
        assert np.all(result[:-1] < result[1:]), f"Result must be sorted, got {result}"
        assert np.all(
            np.isin(indices, result)
        ), f"All samples must be in output, missing: {indices[~np.isin(indices, result)]}"

    def test_trajectory_coverage(self):
        """Test that all required trajectory frames are present.

        For sample at index 100 with min_frames=10:
        - Need frames [91, 92, ..., 99, 100]
        """
        indices = np.array([100])
        min_frames = 10
        result = expand_indices_for_trajectories(indices, min_frames)

        # Should have frames from (100 - min_frames + 1) to 100
        expected_start = 100 - min_frames + 1
        expected_end = 100

        for frame_idx in range(expected_start, expected_end + 1):
            assert (
                frame_idx in result
            ), f"Frame {frame_idx} should be in trajectory for sample 100"

    def test_no_gaps_in_trajectory(self):
        """Test that trajectory windows have no internal gaps."""
        indices = np.array([10])
        result = expand_indices_for_trajectories(indices, 5)

        # Should have exactly 5 consecutive frames
        assert len(result) == 5, f"Expected 5 frames, got {len(result)}: {result}"
        assert np.array_equal(
            result, np.arange(6, 11)
        ), f"Expected [6,7,8,9,10], got {result}"
