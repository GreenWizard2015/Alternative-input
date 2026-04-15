"""Integration performance benchmarks for preprocessing."""

import json
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


class TestMainScriptIntegrationPerformance:
    """Benchmark end-to-end preprocessing pipeline."""

    def test_hierarchical_processing_performance(self, preprocess_module):
        """Benchmark processing of hierarchical folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a simple hierarchy with 3 datasets
            hierarchy_paths = [
                root / "user1" / "screen1" / "camera1" / "monitor1" / "place1",
                root / "user1" / "screen1" / "camera2" / "monitor1" / "place1",
                root / "user2" / "screen1" / "camera1" / "monitor1" / "place1",
            ]

            for path in hierarchy_paths:
                path.mkdir(parents=True, exist_ok=True)
                # Create dataset at leaf
                times = np.linspace(0, 2, 200)
                data = np.random.RandomState(42).randn(200)
                dataset = {"time": times, "data": data}
                np.savez(path / "all.npz", **dataset)

            # Time the main function
            start_time = time.time()

            # Create args-like object
            class Args:
                folder = str(root)
                test_ratio = 0.2
                minimum_frames = 5
                drop_zero_deltas = True
                maxT = 1.0
                random_seed = 42
                filter_threshold = 0.1

            preprocess_module.main(Args())
            elapsed = time.time() - start_time

            # Verify stats file was created
            stats_file = root / "stats.json"
            assert stats_file.exists(), "Stats file should exist"

            with open(stats_file, "r") as f:
                stats = json.load(f)
            assert isinstance(stats, dict), "Stats should be a dictionary"

            # Performance check (3 datasets × 200 frames = 600 total)
            assert (
                elapsed < 30.0
            ), f"Hierarchical processing took {elapsed:.2f}s, should be <30s"
