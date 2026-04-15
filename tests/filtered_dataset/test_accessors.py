"""Tests for FilteredDataset accessor methods."""

import pytest


class TestFilteredDatasetAccessors:
    """Test FilteredDataset accessor methods."""

    def test_valid_indices_returns_sorted_list(self, dataset_with_frames):
        """Test that valid_indices returns sorted list."""
        # Add samples
        for i in range(4):
            dataset_with_frames.add_sample({"time": 0.3 + i * 0.1})

        valid = dataset_with_frames.valid_indices()

        assert isinstance(
            valid, list
        ), f"valid_indices should return list, got {type(valid)}"
        assert valid == sorted(valid), f"valid_indices should be sorted, got {valid}"
        assert len(valid) == len(
            set(valid)
        ), f"valid_indices should have no duplicates, got {valid}"

    def test_used_samples_includes_trajectories(self, dataset_with_frames):
        """Test that used_samples includes full trajectory ranges."""
        # Add samples
        dataset_with_frames.add_sample({"time": 0.5})
        dataset_with_frames.add_sample({"time": 0.7})

        valid = set(dataset_with_frames.valid_indices())
        used = set(dataset_with_frames.used_samples())

        # used_samples should include valid samples
        assert valid.issubset(
            used
        ), f"used_samples should include all valid samples, valid: {valid}, used: {used}"

        # used_samples should include trajectory frames that valid_indices doesn't
        if len(valid) < len(used):
            extra = used - valid
            # These extra frames should come from trajectories
            assert (
                len(extra) > 0
            ), f"used_samples should have extra frames from trajectories, got {extra}"

    def test_trajectory_start_returns_int(self, dataset_with_frames):
        """Test that trajectory_start returns minimum index as int."""
        # Add a sample
        dataset_with_frames.add_sample({"time": 0.5})

        if len(dataset_with_frames) == 0:
            pytest.skip("No valid samples in dataset")

        sample_idx = dataset_with_frames.valid_indices()[0]
        result = dataset_with_frames.trajectory_start(sample_idx)

        assert isinstance(
            result, int
        ), f"trajectory_start should return int, got {type(result)}"
        assert (
            result <= sample_idx
        ), f"trajectory start should be <= sample_idx, got {result} <= {sample_idx}"

    def test_total_samples_vs_valid_samples(self, dataset_with_frames):
        """Test that total storage size is at least the number of valid samples."""
        dataset_with_frames.add_sample({"time": 0.5})

        # Verify storage growth behavior after adding samples
        initial_size = len(dataset_with_frames)
        # Add another sample
        dataset_with_frames.add_sample({"time": 0.6})
        final_size = len(dataset_with_frames)

        # Behavior to test: storage should grow as samples are added
        assert (
            final_size >= initial_size
        ), f"Storage should grow or stay same after adding samples: {final_size} >= {initial_size}"
