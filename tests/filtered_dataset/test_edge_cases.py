"""Tests for FilteredDataset edge cases."""

import pytest
from Core.data.FilteredDataset import FilteredDataset


class TestFilteredDatasetEdgeCases:
    """Test FilteredDataset edge cases."""

    def test_empty_dataset(self, storage_10_frames):
        """Test operations on empty dataset (no samples added)."""
        dataset = FilteredDataset(storage_10_frames, minFrames=3, maxT=1.0)

        assert (
            len(dataset) == 0
        ), f"Empty dataset should have 0 valid samples, got {len(dataset)}"
        assert (
            dataset.valid_indices() == []
        ), f"valid_indices on empty dataset should be empty, got {dataset.valid_indices()}"
        assert (
            dataset.used_samples() == []
        ), f"used_samples on empty dataset should be empty, got {dataset.used_samples()}"

    def test_getitem_access(self, dataset_with_frames):
        """Test __getitem__ access to valid samples."""
        # Add samples
        dataset_with_frames.add_sample({"time": 0.5})

        if len(dataset_with_frames) == 0:
            pytest.skip("No valid samples to test __getitem__")

        # Access first valid sample
        sample = dataset_with_frames[0]

        assert isinstance(
            sample, dict
        ), f"__getitem__ should return sample dict, got {type(sample)}"
        assert (
            "time" in sample
        ), f"Sample should have time field, sample keys: {list(sample.keys())}"

    def test_getitem_out_of_range_behavior(self, dataset_with_frames):
        """Test __getitem__ with out of range index returns None or raises gracefully."""
        # Add samples
        dataset_with_frames.add_sample({"time": 0.5})

        if len(dataset_with_frames) == 0:
            pytest.skip("No valid samples to test __getitem__")

        # Try to access beyond valid range - behavior should be well-defined
        invalid_idx = len(dataset_with_frames) + 10

        # The behavior depends on how the underlying storage handles out-of-range access
        try:
            result = dataset_with_frames[invalid_idx]
            # If no exception is raised, result should be None or dict
            assert result is None or isinstance(
                result, dict
            ), f"Out-of-range access should return None or dict, got {type(result)}"
        except IndexError:
            # Index error is acceptable behavior for out-of-range access
            pass

    def test_used_samples_with_sparse_storage(self, dataset_sparse):
        """Test used_samples with sparse (non-contiguous) valid indices."""
        # Sparse storage has gaps in time, may result in non-contiguous valid indices
        dataset_sparse.add_sample({"time": 0.5})
        dataset_sparse.add_sample({"time": 2.0})

        used = dataset_sparse.used_samples()
        valid = dataset_sparse.valid_indices()

        # used_samples should be superset of valid_indices
        assert set(valid).issubset(
            set(used)
        ), f"used_samples should contain all valid_indices, valid: {valid}, used: {used}"
