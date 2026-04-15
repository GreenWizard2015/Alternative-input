"""Shared fixtures for FilteredDataset tests."""

import pytest

from Core.data.FilteredDataset import FilteredDataset


@pytest.fixture
def dataset_with_frames(storage_10_frames):
    """Create FilteredDataset with mock storage (10 frames at 0.1s intervals)."""
    return FilteredDataset(
        storage_10_frames,
        minFrames=3,
        maxT=1.0,
    )


@pytest.fixture
def dataset_sparse(storage_sparse_frames):
    """Create FilteredDataset with sparse storage."""
    return FilteredDataset(
        storage_sparse_frames,
        minFrames=2,
        maxT=1.0,
    )
