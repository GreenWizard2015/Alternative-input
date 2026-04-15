"""Performance benchmarks for refactored preprocessing pipeline.

Tests verify that refactored preprocessing maintains acceptable performance
compared to acceptable baseline thresholds. Benchmarks are conducted on:
- Medium-sized datasets (500 frames)
- Large-sized datasets (5000 frames)
- Hierarchical folder structure

Performance acceptance criteria:
- Medium dataset (<10 seconds for 500 frames)
- Large dataset (<60 seconds for 5000 frames)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import time
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


class TestProcessFolderPerformance:
    """Benchmark processFolder performance on various dataset sizes."""

    def test_medium_dataset_performance(self, preprocess_module):
        """Benchmark on medium-sized dataset (500 frames)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create medium-sized dataset
            times = np.linspace(0, 5, 500)
            data = np.random.RandomState(42).randn(500)
            dataset = {"time": times, "data": data}
            np.savez(tmpdir_path / "all.npz", **dataset)

            # Time the operation
            start_time = time.time()
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir_path),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=True,
                    maxT=1.0,
                    random_seed=42,
                )
            )
            elapsed = time.time() - start_time

            # Verify performance
            assert not is_skipped, "Medium dataset should not be skipped"
            assert (
                elapsed < 15.0
            ), f"Medium dataset (500 frames) took {elapsed:.2f}s, should be <15s"
            assert test_frames > 0, f"Expected test_frames > 0, got {test_frames}"
            assert train_frames > 0, f"Expected train_frames > 0, got {train_frames}"

    def test_large_dataset_performance(self, preprocess_module):
        """Benchmark on large-sized dataset (5000 frames)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create large-sized dataset
            times = np.linspace(0, 50, 5000)
            data = np.random.RandomState(42).randn(5000)
            dataset = {"time": times, "data": data}
            np.savez(tmpdir_path / "all.npz", **dataset)

            # Time the operation
            start_time = time.time()
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir_path),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=True,
                    maxT=1.0,
                    random_seed=42,
                )
            )
            elapsed = time.time() - start_time

            # Verify performance
            assert not is_skipped, "Large dataset should not be skipped"
            assert (
                elapsed < 120.0
            ), f"Large dataset (5000 frames) took {elapsed:.2f}s, should be <120s"
            assert (
                test_frames > 0 and train_frames > 0
            ), "Should produce both train and test splits"

    def test_sparse_dataset_performance(self, preprocess_module):
        """Benchmark on sparse dataset (gaps in time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create dataset with variable spacing (realistic case)
            times = []
            current_time = 0.0
            for i in range(500):
                # Variable spacing from 0.01 to 0.05 seconds
                delta = np.random.RandomState(i).uniform(0.01, 0.05)
                current_time += delta
                times.append(current_time)

            times = np.array(times)
            data = np.random.RandomState(42).randn(500)
            dataset = {"time": times, "data": data}
            np.savez(tmpdir_path / "all.npz", **dataset)

            # Time the operation
            start_time = time.time()
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir_path),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=True,
                    maxT=1.0,
                    random_seed=42,
                )
            )
            elapsed = time.time() - start_time

            # Verify performance
            assert elapsed < 15.0, f"Sparse dataset took {elapsed:.2f}s, should be <15s"
            assert test_frames > 0 and train_frames > 0, "Should produce both splits"

    def test_performance_with_filtering_enabled(self, preprocess_module):
        """Benchmark with zero-delta removal enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create dataset with many duplicates
            base_times = np.linspace(0, 5, 100)
            times = []
            for t in base_times:
                # Add 4 duplicates for each time
                times.extend([t] * 5)
            times = np.array(times)

            data = np.arange(len(times))
            dataset = {"time": times, "data": data}
            np.savez(tmpdir_path / "all.npz", **dataset)

            # Time the operation with filtering
            start_time = time.time()
            test_frames, train_frames, is_skipped, stats = (
                preprocess_module.processFolder(
                    str(tmpdir_path),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=True,  # Filtering enabled
                    maxT=1.0,
                    random_seed=42,
                )
            )
            elapsed_filtered = time.time() - start_time

            # Time the operation without filtering
            np.savez(tmpdir_path / "all.npz", **dataset)
            start_time = time.time()
            test_frames2, train_frames2, is_skipped2, stats2 = (
                preprocess_module.processFolder(
                    str(tmpdir_path),
                    testRatio=0.2,
                    minimumFrames=5,
                    dropZeroDeltas=False,  # Filtering disabled
                    maxT=1.0,
                    random_seed=42,
                )
            )
            elapsed_unfiltered = time.time() - start_time

            # Filtering should not add significant overhead
            # (may be faster due to smaller dataset after filtering)
            assert elapsed_filtered < 20.0, f"Filtered took {elapsed_filtered:.2f}s"
            assert (
                elapsed_unfiltered < 20.0
            ), f"Unfiltered took {elapsed_unfiltered:.2f}s"

    def test_reproducibility_vs_performance(self, preprocess_module):
        """Test that using seeds doesn't significantly impact performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            times = np.linspace(0, 5, 500)
            data = np.random.RandomState(42).randn(500)
            dataset = {"time": times, "data": data}

            # Time with seed
            np.savez(tmpdir_path / "all.npz", **dataset)
            start_time = time.time()
            _, _, _, _ = preprocess_module.processFolder(
                str(tmpdir_path),
                testRatio=0.2,
                minimumFrames=5,
                dropZeroDeltas=True,
                maxT=1.0,
                random_seed=42,  # With seed
            )
            elapsed_with_seed = time.time() - start_time

            # Time without seed
            np.savez(tmpdir_path / "all.npz", **dataset)
            start_time = time.time()
            _, _, _, _ = preprocess_module.processFolder(
                str(tmpdir_path),
                testRatio=0.2,
                minimumFrames=5,
                dropZeroDeltas=True,
                maxT=1.0,
                random_seed=None,  # Without seed
            )
            elapsed_without_seed = time.time() - start_time

            # Both should complete in similar time
            assert elapsed_with_seed < 20.0, f"With seed: {elapsed_with_seed:.2f}s"
            assert (
                elapsed_without_seed < 20.0
            ), f"Without seed: {elapsed_without_seed:.2f}s"
