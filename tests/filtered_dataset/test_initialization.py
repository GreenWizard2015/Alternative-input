"""Tests for FilteredDataset initialization and validation."""

import pytest
from Core.data.FilteredDataset import FilteredDataset


class TestFilteredDatasetInitialization:
    """Test FilteredDataset initialization and validation."""

    def test_valid_initialization(self, storage_10_frames):
        """Test FilteredDataset initializes with valid parameters."""
        _ = FilteredDataset(storage_10_frames, minFrames=3, maxT=1.0)
        # Initialization succeeds without raising exceptions

    def test_initialization_zero_min_frames_prevented(self, storage_10_frames):
        """Test that minFrames=0 prevents normal dataset creation - behavior test.

        The behavior is that zero minFrames is invalid and results in the dataset
        not being created properly (whether by exception or returning None).
        """
        # Attempt to create dataset with invalid parameters
        with pytest.raises(ValueError, match="minFrames must be positive"):
            FilteredDataset(storage_10_frames, minFrames=0, maxT=1.0)

    def test_initialization_negative_min_frames_prevented(self, storage_10_frames):
        """Test that negative minFrames prevents normal dataset creation - behavior test.

        The behavior is that negative minFrames is invalid and results in the dataset
        not being created properly (whether by exception or returning None).
        """
        # Attempt to create dataset with invalid parameters
        with pytest.raises(ValueError, match="minFrames must be positive"):
            FilteredDataset(storage_10_frames, minFrames=-1, maxT=1.0)

    def test_initialization_min_frames_affects_validity(self, storage_10_frames):
        """Test that minFrames parameter affects which samples are valid.

        More restrictive minFrames results in fewer valid samples.
        """
        dataset_1 = FilteredDataset(storage_10_frames, minFrames=1, maxT=0.05)
        dataset_5 = FilteredDataset(storage_10_frames, minFrames=5, maxT=0.05)

        # Add same samples to both
        dataset_1.add_sample({"time": 0.15})
        dataset_5.add_sample({"time": 0.15})

        # With minFrames=1 and small maxT, more samples should be valid
        count_1 = len(dataset_1)

        # With minFrames=5 and small maxT, fewer samples should be valid (stricter)
        count_5 = len(dataset_5)

        assert (
            count_1 >= count_5
        ), f"minFrames=1 should have >= valid samples than minFrames=5, got {count_1} vs {count_5}"

    def test_initialization_max_t_affects_trajectory_range(self, storage_10_frames):
        """Test that maxT parameter affects trajectory ranges."""
        dataset_small = FilteredDataset(storage_10_frames, minFrames=1, maxT=0.1)
        dataset_large = FilteredDataset(storage_10_frames, minFrames=1, maxT=2.0)

        # Add sample at same position to both
        dataset_small.add_sample({"time": 0.5})
        dataset_large.add_sample({"time": 0.5})

        # Get trajectory start indices for first valid sample
        small_idx = dataset_small.valid_indices()[0]
        large_idx = dataset_large.valid_indices()[0]

        min_small = dataset_small.trajectory_start(small_idx)
        min_large = dataset_large.trajectory_start(large_idx)

        range_small = small_idx - min_small
        range_large = large_idx - min_large

        # Larger maxT should produce larger trajectory ranges
        assert (
            range_large >= range_small
        ), f"Larger maxT should give larger range: {range_large} >= {range_small}"
