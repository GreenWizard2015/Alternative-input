"""Tests for DatasetPreprocessor index expansion and trajectory methods."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestExpandSampleIndices:
    """Test expand_sample_indices() method."""

    def test_basic_index_expansion(self):
        """Test expanding sample indices to frame indices."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([10, 20, 30])
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=100
        )

        # For sample at 10, need frames [6, 7, 8, 9, 10]
        # For sample at 20, need frames [16, 17, 18, 19, 20]
        # For sample at 30, need frames [26, 27, 28, 29, 30]
        assert 10 in frame_indices, "10 should be in frame_indices"
        assert 20 in frame_indices, "20 should be in frame_indices"
        assert 30 in frame_indices, "30 should be in frame_indices"
        assert (
            6 in frame_indices
        ), "Frame 6 should be in result (min frame for sample 10)"
        assert (
            26 in frame_indices
        ), "Frame 26 should be in result (min frame for sample 30)"

    def test_expanded_indices_sorted(self):
        """Test that expanded indices are sorted."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([10, 20, 30])
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=100
        )

        assert np.all(
            frame_indices[:-1] < frame_indices[1:]
        ), "Expanded indices should be strictly sorted"

    def test_expanded_indices_no_duplicates(self):
        """Test that expanded indices have no duplicates."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([10, 12, 14])  # Close together
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=50
        )

        unique_count = len(np.unique(frame_indices))
        assert unique_count == len(
            frame_indices
        ), "Expanded indices should have no duplicates"

    def test_sample_too_close_to_start_raises(self):
        """Test that sample too close to dataset start raises."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([3])  # Need frames [−1, 0, 1, 2, 3]
        with pytest.raises(ValueError, match="minimum"):
            preprocessor.expand_sample_indices(sample_indices, dataset_size=100)

    def test_sample_at_minimum_valid_position(self):
        """Test sample at minimum valid position (min_trajectory_frames - 1)."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([4])  # Needs frames [0, 1, 2, 3, 4]
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=100
        )

        assert 0 in frame_indices, "0 should be in frame_indices"
        assert 4 in frame_indices, "4 should be in frame_indices"

    def test_sample_beyond_dataset_raises(self):
        """Test that sample beyond dataset size raises."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([100])  # Beyond dataset_size=50
        with pytest.raises(ValueError, match="maximum"):
            preprocessor.expand_sample_indices(sample_indices, dataset_size=50)

    def test_single_sample_expansion(self):
        """Test expansion with single sample."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=3)

        sample_indices = np.array([10])
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=50
        )

        expected = np.array([8, 9, 10])  # [10 - 3 + 1, ..., 10]
        assert np.array_equal(frame_indices, expected), "Arrays should be equal"

    def test_multiple_samples_coverage(self):
        """Test that all samples are represented in expanded indices."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([5, 15, 25, 35])
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=50
        )

        for sample_idx in sample_indices:
            assert (
                sample_idx in frame_indices
            ), f"Sample {sample_idx} should be in expanded indices"

    def test_overlapping_trajectories_merged(self):
        """Test that overlapping trajectory windows are merged correctly."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        # Samples [5, 6, 7] have overlapping trajectory windows
        # [5] needs [1, 2, 3, 4, 5]
        # [6] needs [2, 3, 4, 5, 6]
        # [7] needs [3, 4, 5, 6, 7]
        # Union should be [1, 2, 3, 4, 5, 6, 7]
        sample_indices = np.array([5, 6, 7])
        frame_indices = preprocessor.expand_sample_indices(
            sample_indices, dataset_size=50
        )

        expected_count = 7  # [1, 2, 3, 4, 5, 6, 7]
        assert (
            len(frame_indices) == expected_count
        ), f"Expected {expected_count} frames from overlapping windows, got {len(frame_indices)}"
