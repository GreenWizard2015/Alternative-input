"""Tests for expand_indices wrapper function."""

import numpy as np
import pytest
import importlib.util


@pytest.fixture
def preprocess_module():
    """Load the refactored preprocess-remote.py script."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_remote", "scripts/preprocess-remote.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRefactoredExpandIndices:
    """Test expand_indices wrapper function."""

    def test_wrapper_expands_correctly(self, preprocess_module):
        """Test that wrapper expands indices correctly."""
        indices = np.array([10, 20])
        min_frames = 5

        result = preprocess_module.expand_indices(indices, min_frames)

        expected = np.array([6, 7, 8, 9, 10, 16, 17, 18, 19, 20])
        assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    def test_wrapper_returns_sorted(self, preprocess_module):
        """Test that wrapper returns sorted results."""
        indices = np.array([30, 10, 20])  # Unsorted input
        min_frames = 3

        result = preprocess_module.expand_indices(indices, min_frames)

        # Should be sorted
        assert np.all(
            result[:-1] < result[1:]
        ), f"Result should be sorted, got {result}"

    def test_wrapper_no_duplicates(self, preprocess_module):
        """Test that wrapper output has no duplicates."""
        indices = np.array([5, 6, 7])  # Close together
        min_frames = 5

        result = preprocess_module.expand_indices(indices, min_frames)

        unique_count = len(np.unique(result))
        assert unique_count == len(result), "Output should have no duplicates"
