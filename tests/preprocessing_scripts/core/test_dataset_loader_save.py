"""Tests for DatasetLoader save functionality."""

import numpy as np
from pathlib import Path
import tempfile
import pytest
import time
from scripts.preprocessing.traversal import DatasetLoader


class TestDatasetLoaderSave:
    """Test DatasetLoader save functionality."""

    def test_save_dataset(self):
        """Test saving dataset to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            dataset = {"time": np.array([0.0, 0.1, 0.2]), "data": np.array([1, 2, 3])}

            loader = DatasetLoader(tmpdir)
            loader.save_dataset(dataset, output_dir)

            assert (
                output_dir / "time.npy"
            ).exists(), "Expected time.npy to exist in output directory"
            assert (
                output_dir / "data.npy"
            ).exists(), "Expected data.npy to exist in output directory"

    def test_save_overwrites_with_flag(self):
        """Test that overwrite flag controls file replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            dataset = {"data": np.array([1, 2, 3])}

            loader = DatasetLoader(tmpdir)

            # First save
            loader.save_dataset(dataset, tmpdir, overwrite=False)
            first_mtime = (tmpdir / "data.npy").stat().st_mtime

            # Try to save again without overwrite
            time.sleep(0.01)  # Ensure different timestamp
            with pytest.raises(FileExistsError):
                loader.save_dataset(dataset, tmpdir, overwrite=False)

            # With overwrite should work
            loader.save_dataset(dataset, tmpdir, overwrite=True)
            second_mtime = (tmpdir / "data.npy").stat().st_mtime

            assert (
                second_mtime > first_mtime
            ), f"Expected second mtime ({second_mtime}) > first mtime ({first_mtime})"

    def test_save_skips_non_arrays(self):
        """Test that non-array values are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            dataset = {
                "data": np.array([1, 2, 3]),
                "metadata": "string_value",
                "count": 42,
            }

            loader = DatasetLoader(tmpdir)
            loader.save_dataset(dataset, tmpdir)

            # Only data.npy should exist
            assert (tmpdir / "data.npy").exists(), "Expected data.npy to exist"
            assert not (
                tmpdir / "metadata.npy"
            ).exists(), "Expected metadata.npy to not exist (non-array)"
