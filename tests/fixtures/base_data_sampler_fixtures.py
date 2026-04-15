"""Shared fixtures for BaseDataSampler tests."""

import pytest

from Core.data.BaseDataSampler import BaseDataSampler
from tests.fixtures.mock_storage import _MockStorage


@pytest.fixture
def sampler_with_frames():
    """Create BaseDataSampler with mock storage (10 frames at 0.1s intervals)."""
    times = [i * 0.1 for i in range(10)]
    storage = _MockStorage(times)
    return BaseDataSampler(
        storage=storage,
        batch_size=32,
        minFrames=3,
        defaults={"timesteps": 5},
        maxT=1.0,
    )
