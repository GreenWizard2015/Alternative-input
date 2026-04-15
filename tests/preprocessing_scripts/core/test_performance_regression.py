"""Performance regression tests."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import importlib.util
import time


@pytest.fixture
def preprocess_module():
    """Load the refactored preprocess-remote.py script."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_remote", "scripts/preprocess-remote.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPerformanceRegression:
    """Track performance metrics across runs."""

    def test_performance_metrics_logged(self, preprocess_module):
        """Verify performance is within acceptable bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create realistic dataset
            times = np.linspace(0, 10, 1000)
            data = np.random.RandomState(42).randn(1000)
            dataset = {"time": times, "data": data}
            np.savez(tmpdir_path / "all.npz", **dataset)

            # Collect metrics
            metrics = {
                "dataset_size": 1000,
                "elapsed_time": None,
                "test_frames": None,
                "train_frames": None,
                "frames_per_second": None,
            }

            start_time = time.time()
            test_frames, train_frames, is_skipped, _ = preprocess_module.processFolder(
                str(tmpdir_path),
                testRatio=0.2,
                minimumFrames=5,
                dropZeroDeltas=True,
                maxT=1.0,
                random_seed=42,
            )
            elapsed = time.time() - start_time

            metrics["elapsed_time"] = elapsed
            metrics["test_frames"] = test_frames
            metrics["train_frames"] = train_frames
            metrics["frames_per_second"] = 1000 / elapsed if elapsed > 0 else 0

            # Acceptance criteria
            assert not is_skipped, "Dataset should not be skipped"
            assert elapsed < 30.0, f"Processing took {elapsed:.2f}s, should be <30s"
            assert (
                metrics["frames_per_second"] > 33
            ), f"Processing speed {metrics['frames_per_second']:.1f} fps is too slow"
            assert metrics["elapsed_time"] > 0, "Elapsed time should be positive"
            assert (
                metrics["test_frames"] >= 0
            ), "Test frames count should be non-negative"
            assert (
                metrics["train_frames"] >= 0
            ), "Train frames count should be non-negative"
