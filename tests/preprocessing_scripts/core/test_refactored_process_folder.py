"""Tests for refactored processFolder function."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
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


class TestRefactoredProcessFolder:
    """Test that processFolder still works with refactored utility functions."""

    def test_basic_folder_processing(self, preprocess_module):
        """Test basic folder processing with refactored functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple dataset
            times = np.linspace(0, 1, 50)
            data = np.arange(50) * 10
            dataset = {"time": times, "data": data}

            # Save as all.npz
            np.savez(tmpdir / "all.npz", **dataset)

            # Create a mock "npz file" to be removed
            np.savez(tmpdir / "old.npz", dummy=np.array([1, 2, 3]))

            # Process the folder
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=True,
                    maxT=1.0,
                    random_seed=42,
                )
            )

            # Should not be skipped and should have results
            assert not is_skipped, "Dataset should not be skipped"
            assert test_frames > 0, f"Test frames should be > 0, got {test_frames}"
            assert train_frames > 0, f"Train frames should be > 0, got {train_frames}"
            assert (
                test_frames + train_frames > 0
            ), f"Total frames should be > 0, got {test_frames + train_frames}"

    def test_folder_processing_with_sparse_data(self, preprocess_module):
        """Test that sparse data is properly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sparse dataset (large gaps)
            times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
            data = np.arange(5)
            dataset = {"time": times, "data": data}

            np.savez(tmpdir / "all.npz", **dataset)

            # Process the folder
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir),
                    testRatio=0.2,
                    minimumFrames=3,
                    dropZeroDeltas=True,
                    maxT=1.0,
                    random_seed=42,
                )
            )

            # Should be skipped due to sparsity
            assert is_skipped, "Sparse dataset should be skipped"
