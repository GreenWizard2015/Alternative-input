"""Tests for expand_indices_for_trajectories() edge cases and boundary conditions."""

import pytest
import numpy as np
from scripts.preprocessing.core import expand_indices_for_trajectories


class TestExpandIndicesEdgeCases:
    """Test expand_indices_for_trajectories() boundary and edge cases."""

    def test_sample_near_zero_boundary(self):
        """Test sample too close to boundary (negative trajectory expansion).

        Sample at index 3 with min_frames=5 would need frames [-2,-1,0,1,2,3]
        Should raise ValueError for negative indices.
        """
        indices = np.array([3])  # Close to boundary
        min_frames = 5

        with pytest.raises(ValueError, match="negative"):
            expand_indices_for_trajectories(indices, min_frames)

    def test_sample_at_boundary_zero(self):
        """Test sample at index 0 with min_frames > 1.

        Sample at 0 with min_frames=5 needs frames [-4,-3,-2,-1,0]
        Should raise ValueError.
        """
        indices = np.array([0])

        with pytest.raises(ValueError, match="negative"):
            expand_indices_for_trajectories(indices, 5)

    def test_sample_just_at_min_frames_boundary(self):
        """Test sample at exactly min_frames (boundary case).

        Sample at index 5 with min_frames=5 needs frames [1,2,3,4,5]
        This should work (all non-negative).
        """
        indices = np.array([5])
        result = expand_indices_for_trajectories(indices, 5)

        expected = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(result, expected)

    def test_sample_at_min_frames_minus_1(self):
        """Test sample just before minimum boundary.

        Sample at index 4 with min_frames=5 needs frames [0,1,2,3,4]
        This should work.
        """
        indices = np.array([4])
        result = expand_indices_for_trajectories(indices, 5)

        expected = np.array([0, 1, 2, 3, 4])
        assert np.array_equal(result, expected)

    def test_min_frames_equals_sample_index(self):
        """Test when min_frames == sample_index.

        Sample at 5 with min_frames=5 needs frames [1-5]
        """
        indices = np.array([5])
        result = expand_indices_for_trajectories(indices, 5)

        assert 0 not in result, "Frame 0 should not be in trajectory"
        assert 1 in result, "Trajectory should start at 1"
        assert 5 in result, "Sample should be in result"

    def test_mixed_boundary_and_safe_samples(self):
        """Test mix of samples, some near boundary, some safe.

        [2, 100] with min_frames=5:
        - Sample 2 would need [-3,-2,-1,0,1,2] → ERROR
        """
        indices = np.array([2, 100])

        with pytest.raises(ValueError):
            expand_indices_for_trajectories(indices, 5)

    def test_very_large_indices(self):
        """Test with very large sample indices."""
        indices = np.array([1000000])
        result = expand_indices_for_trajectories(indices, 5)

        assert 999996 in result, "Trajectory should start"
        assert 1000000 in result, "Sample should be in result"
        assert len(result) == 5, "Should have exactly 5 frames"

    def test_unsorted_input_indices(self):
        """Test that function handles unsorted input correctly.

        Even if input is unsorted, output must be sorted.
        """
        indices = np.array([100, 10, 50])  # Unsorted
        result = expand_indices_for_trajectories(indices, 5)

        # Must be sorted
        assert np.all(result[:-1] < result[1:]), f"Output must be sorted, got {result}"

    def test_duplicate_input_indices(self):
        """Test behavior with duplicate indices in input.

        expand_indices should handle duplicates via np.unique()
        """
        indices = np.array([50, 50, 50])  # All same
        result = expand_indices_for_trajectories(indices, 5)

        # np.unique() should eliminate duplicates
        unique_result = np.unique(result)
        assert len(result) == len(unique_result), "Result should not have duplicates"

    def test_consecutive_samples(self):
        """Test with consecutive sample indices.

        Samples [10, 11, 12] with min_frames=5:
        Trajectories [6-10], [7-11], [8-12]
        Union: [6,7,8,9,10,11,12]
        """
        indices = np.array([10, 11, 12])
        result = expand_indices_for_trajectories(indices, 5)

        # All trajectory windows overlap heavily
        expected_min = 10 - 5 + 1  # 6
        expected_max = 12  # 12

        assert (
            np.min(result) == expected_min
        ), f"Min should be {expected_min}, got {np.min(result)}"
        assert (
            np.max(result) == expected_max
        ), f"Max should be {expected_max}, got {np.max(result)}"
        assert len(result) == 7, "Should have 7 frames [6-12]"

    def test_spaced_samples_max_min_frames(self):
        """Test samples maximally spaced (no overlap).

        Samples [10, 100] with min_frames=5:
        Trajectories [6-10] and [96-100]
        No overlap.
        """
        indices = np.array([10, 100])
        result = expand_indices_for_trajectories(indices, 5)

        # Check no overlap
        first_traj = set(range(6, 11))
        second_traj = set(range(96, 101))

        assert first_traj.isdisjoint(second_traj), "Trajectories should not overlap"

        # All frames from both should be present
        assert first_traj.issubset(set(result)), "First trajectory not fully in result"
        assert second_traj.issubset(
            set(result)
        ), "Second trajectory not fully in result"

    def test_return_array_properties(self):
        """Test properties of returned array."""
        indices = np.array([10, 20, 30])
        result = expand_indices_for_trajectories(indices, 5)

        # Must be numpy array
        assert isinstance(result, np.ndarray), "Result must be numpy array"

        # Must have integer dtype
        assert result.dtype in [
            np.int32,
            np.int64,
        ], f"Result must have integer dtype, got {result.dtype}"

        # Must be 1D
        assert result.ndim == 1, f"Result must be 1D array, got shape {result.shape}"

    def test_memory_efficiency(self):
        """Test that function doesn't create unnecessary large arrays.

        Should not allocate arrays much larger than needed.
        """
        indices = np.array([10, 20, 30])
        result = expand_indices_for_trajectories(indices, 5)

        # With 3 samples and 5 frames each, max is 15 frames (no overlap)
        # With overlap, less. But should not exceed 20.
        assert len(result) <= 20, f"Result too large: {len(result)} (expected <= 20)"
