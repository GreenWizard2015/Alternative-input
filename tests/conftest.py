"""Pytest configuration and fixture discovery."""

from Core.logging_config import setup_root_logger
from tests.fixtures.embedding_fixtures import (
    stats_data,
    cached_model_wrappers,
    shared_test_inputs,
)
from tests.fixtures.base_data_sampler_fixtures import sampler_with_frames
from tests.fixtures.sample_filter_fixtures import (
    storage_10_frames,
    storage_sparse_frames,
    storage_single_frame,
    filter_with_1s_window,
)
from tests.fixtures.filtered_dataset_fixtures import (
    dataset_with_frames,
    dataset_sparse,
)
from tests.fixtures.evaluation_tracker_fixtures import (
    mock_model,
    mock_datasets,
)
from tests.fixtures.test_inputs import create_test_inputs

__all__ = [
    "stats_data",
    "cached_model_wrappers",
    "shared_test_inputs",
    "sampler_with_frames",
    "storage_10_frames",
    "storage_sparse_frames",
    "storage_single_frame",
    "filter_with_1s_window",
    "dataset_with_frames",
    "dataset_sparse",
    "mock_model",
    "mock_datasets",
    "create_test_inputs",
]


def pytest_configure(config):
    """Configure global logging for all tests and modules."""
    setup_root_logger(suppress_third_party=True)
