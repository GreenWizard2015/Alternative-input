"""Performance baseline tests for regression testing."""

import numpy as np
from pathlib import Path
import tempfile
import importlib.util
import time
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


class TestPerformanceBaseline:
    """Establish performance baseline for regression testing."""

    def test_processFolder_timing_baseline(self, preprocess_module):
        """Establish timing baseline for processFolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a medium-sized dataset
            times = np.linspace(0, 5, 500)
            data = np.random.RandomState(42).randn(500)
            dataset = {"time": times, "data": data}

            np.savez(tmpdir_path / "all.npz", **dataset)

            # Time the operation
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

            # Should complete in reasonable time (<10 seconds for 500 frames)
            assert elapsed < 10.0, "Processing should complete in <10 seconds"

            # Should not be skipped
            assert not is_skipped, "Dataset should not be skipped"
