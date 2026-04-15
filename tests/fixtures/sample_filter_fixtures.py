"""Shared fixtures for SampleFilter tests."""

import pytest

from Core.data.SampleFilter import SampleFilter
from tests.fixtures.mock_storage import _MockStorage


@pytest.fixture
def storage_10_frames():
    """Create mock storage with 10 frames at 0.1s intervals (1 second total)."""
    times = [i * 0.1 for i in range(10)]
    return _MockStorage(times)


@pytest.fixture
def storage_sparse_frames():
    """Create mock storage with sparse frames (larger time gaps)."""
    times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 5.0]
    return _MockStorage(times)


@pytest.fixture
def storage_single_frame():
    """Create mock storage with single frame."""
    return _MockStorage([0.0])


@pytest.fixture
def filter_with_1s_window(storage_10_frames):
    """Create SampleFilter with 1 second time window and 3 minimum frames."""
    return SampleFilter(storage_10_frames, minFrames=3, maxT=1.0)
