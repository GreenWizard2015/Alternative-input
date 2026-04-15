"""Tests for compute_sample_frame_mapping method."""

import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestComputeSampleFrameMapping:
    """Test compute_sample_frame_mapping() method."""

    def test_basic_frame_mapping(self):
        """Test computing frame ranges for samples."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([10, 20, 30])
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        # For min_trajectory_frames=5:
        # Sample 10: frames [6, 11) = [6, 7, 8, 9, 10]
        # Sample 20: frames [16, 21) = [16, 17, 18, 19, 20]
        # Sample 30: frames [26, 31) = [26, 27, 28, 29, 30]
        assert mapping[10] == (6, 11), "mapping[10] should equal (6, 11)"
        assert mapping[20] == (16, 21), "mapping[20] should equal (16, 21)"
        assert mapping[30] == (26, 31), "mapping[30] should equal (26, 31)"

    def test_mapping_contains_all_samples(self):
        """Test that mapping contains all sample indices."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([5, 10, 15, 20])
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        assert len(mapping) == len(
            sample_indices
        ), "Mapping should have same length as sample_indices"
        for sample_idx in sample_indices:
            assert sample_idx in mapping, "sample_idx should be in mapping"

    def test_frame_range_syntax(self):
        """Test that frame ranges use [start, end) (end is exclusive)."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=3)

        sample_indices = np.array([5])
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        start, end = mapping[5]
        assert end - start == 3, "end - start should equal 3"

    def test_single_sample_mapping(self):
        """Test mapping with single sample."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=7)

        sample_indices = np.array([10])
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        start, end = mapping[10]
        assert start == 4  # 10 - 7 + 1
        assert end == 11  # 10 + 1

    def test_empty_sample_indices(self):
        """Test mapping with empty sample indices."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([], dtype=np.int64)
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        assert len(mapping) == 0, "Mapping should be empty for empty input"

    def test_mapping_boundary_frames(self):
        """Test that mapping correctly handles boundary frames."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        sample_indices = np.array([4])  # Minimum valid for min_trajectory_frames=5
        mapping = preprocessor.compute_sample_frame_mapping(sample_indices)

        start, end = mapping[4]
        assert start == 0, "start should equal 0 for boundary frame"
        assert end == 5, "end should equal 5 for boundary frame"
