"""Integration tests verifying refactored preprocess-remote.py behavior.

These tests verify that the refactored main script produces identical behavior
to the original implementation when using the new preprocessing modules.
"""

import pytest
import numpy as np
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


class TestRefactoredDropZeroTimeDelta:
    """Test dropZeroTimeDelta wrapper function."""

    def test_wrapper_removes_duplicates(self, preprocess_module):
        """Test that the wrapper correctly removes duplicate timestamps."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2, 0.3]),
            "data": np.array([1, 2, 3, 4, 5]),
        }

        result = preprocess_module.dropZeroTimeDelta(dataset)

        assert (
            len(result["time"]) == 4
        ), f"Expected 4 timestamps after removing duplicates, got {len(result['time'])}"
        assert np.array_equal(
            result["data"], np.array([1, 2, 4, 5])
        ), f"Expected data [1,2,4,5], got {result['data']}"

    def test_wrapper_preserves_all_fields(self, preprocess_module):
        """Test that wrapper preserves all dataset fields."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.2]),
            "data": np.array([1, 2, 3, 4]),
            "landmarks": np.array([10, 20, 30, 40]),
            "extra": np.array([100, 200, 300, 400]),
        }

        result = preprocess_module.dropZeroTimeDelta(dataset)

        assert "time" in result, "Result should contain 'time' key"
        assert "data" in result, "Result should contain 'data' key"
        assert "landmarks" in result, "Result should contain 'landmarks' key"
        assert "extra" in result, "Result should contain 'extra' key"
        assert all(
            len(v) == 3 for v in result.values() if isinstance(v, np.ndarray)
        ), f"All arrays should have length 3, got lengths: {[len(v) for v in result.values() if isinstance(v, np.ndarray)]}"

    def test_wrapper_converges(self, preprocess_module):
        """Test that wrapper converges with multiple duplicates."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3]),
            "data": np.arange(7),
        }

        result = preprocess_module.dropZeroTimeDelta(dataset)

        assert (
            len(result["time"]) == 4
        ), f"Expected 4 timestamps, got {len(result['time'])}"
        assert np.allclose(
            result["time"], [0.0, 0.1, 0.2, 0.3]
        ), f"Expected times [0.0, 0.1, 0.2, 0.3], got {result['time']}"
