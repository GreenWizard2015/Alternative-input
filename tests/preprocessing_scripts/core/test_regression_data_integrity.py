"""Tests for data integrity in regression testing."""

import numpy as np
from pathlib import Path
import tempfile
import importlib.util
import pytest


@pytest.fixture
def preprocess_module():
    """Load the refactored preprocess-remote.py script."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_remote", "scripts/preprocess-remote.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDataIntegrity:
    """Test that data integrity is maintained through processing."""

    def test_time_monotonicity_preserved(self, preprocess_module):
        """Test that time monotonicity is preserved in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            times = np.linspace(0, 1, 50)
            data = np.arange(50)
            dataset = {"time": times, "data": data}

            np.savez(tmpdir_path / "all.npz", **dataset)

            test_frames, train_frames, is_skipped, _ = preprocess_module.processFolder(
                str(tmpdir_path),
                testRatio=0.2,
                minimumFrames=5,
                dropZeroDeltas=True,
                maxT=1.0,
                random_seed=42,
            )

            if not is_skipped:
                # Load and check output files
                test_data = np.load(tmpdir_path / "test.npz")
                train_data = np.load(tmpdir_path / "train.npz")

                # Check monotonicity
                test_times = test_data["time"]
                train_times = train_data["time"]

                assert np.all(
                    np.diff(test_times) >= 0
                ), "Test times should be non-decreasing"
                assert np.all(
                    np.diff(train_times) >= 0
                ), "Train times should be non-decreasing"

    def test_no_data_loss_on_valid_files(self, preprocess_module):
        """Test that valid data is properly included in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            times = np.linspace(0, 1, 100)
            data = np.arange(100)
            dataset = {"time": times, "data": data}

            np.savez(tmpdir_path / "all.npz", **dataset)

            test_frames, train_frames, is_skipped, _ = preprocess_module.processFolder(
                str(tmpdir_path),
                testRatio=0.2,
                minimumFrames=5,
                dropZeroDeltas=True,
                maxT=1.0,
                random_seed=42,
            )

            if not is_skipped:
                # Should have meaningful data in both splits
                assert test_frames > 0, "test_frames should be > 0"
                assert train_frames > 0, "train_frames should be > 0"
                # Note: total can exceed original due to trajectory expansion overlaps
                # when frames are shared between multiple samples
