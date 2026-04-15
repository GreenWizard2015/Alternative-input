"""Tests for DatasetLoader basic functionality."""

import numpy as np
from pathlib import Path
import tempfile
import pytest
from scripts.preprocessing.traversal import DatasetLoader


class TestDatasetLoaderBasic:
    """Test DatasetLoader basic functionality."""

    def test_loader_initialization(self):
        """Test DatasetLoader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(Path(tmpdir))
            # Test that loader can process the directory (behavior test instead of private field access)
            result = loader.load_dataset(Path(tmpdir))
            # If directory is valid and empty, should return None
            assert result is None, f"Expected None for empty directory, got {result}"

    def test_nonexistent_directory_warning(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        # Should raise FileNotFoundError
        loader = DatasetLoader(Path("/nonexistent/path"))
        with pytest.raises(FileNotFoundError, match="Folder does not exist"):
            loader.load_dataset(Path("/nonexistent/path"))

    def test_load_dataset_empty_folder(self):
        """Test loading from empty folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(Path(tmpdir))
            dataset = loader.load_dataset(Path(tmpdir))

            assert dataset is None, f"Expected None for empty folder, got {dataset}"

    def test_load_dataset_with_npy_files(self):
        """Test loading dataset with .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create .npy files
            np.save(tmpdir / "time.npy", np.array([0.0, 0.1, 0.2]))
            np.save(tmpdir / "data.npy", np.array([1, 2, 3]))

            loader = DatasetLoader(tmpdir)
            dataset = loader.load_dataset(tmpdir)

            assert dataset is not None, f"Expected dataset to be loaded, got {dataset}"
            assert (
                "time" in dataset
            ), f"Expected 'time' key in dataset, got keys: {list(dataset.keys())}"
            assert (
                "data" in dataset
            ), f"Expected 'data' key in dataset, got keys: {list(dataset.keys())}"
            assert (
                len(dataset["time"]) == 3
            ), f"Expected 3 time samples, got {len(dataset['time'])} samples"

    def test_load_dataset_ignores_non_npy(self):
        """Test that non-.npy files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            np.save(tmpdir / "data.npy", np.array([1, 2, 3]))
            (tmpdir / "readme.txt").write_text("ignored")

            loader = DatasetLoader(tmpdir)
            dataset = loader.load_dataset(tmpdir)

            assert dataset is not None, f"Expected dataset to be loaded, got {dataset}"
            assert (
                "data" in dataset
            ), f"Expected 'data' key in dataset, got keys: {list(dataset.keys())}"
            assert (
                len(dataset) == 1
            ), f"Expected 1 item in dataset (only .npy files), got {len(dataset)} items: {list(dataset.keys())}"
