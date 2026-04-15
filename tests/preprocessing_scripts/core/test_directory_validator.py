"""Tests for DirectoryValidator utility methods."""

import numpy as np
from pathlib import Path
import tempfile
from scripts.preprocessing.traversal import DirectoryValidator


class TestDirectoryValidator:
    """Test DirectoryValidator utility methods."""

    def test_is_leaf_directory_no_subdirs(self):
        """Test that directory without subdirectories is a leaf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            result = DirectoryValidator.is_leaf_directory(tmpdir)
            assert (
                result is True
            ), f"Expected directory without subdirs to be leaf, got result: {result} (type: {type(result)})"

    def test_is_leaf_directory_with_subdirs(self):
        """Test that directory with subdirectories is not a leaf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "subdir").mkdir()
            result = DirectoryValidator.is_leaf_directory(tmpdir)
            assert (
                result is False
            ), f"Expected directory with subdirs to not be leaf, got result: {result} (type: {type(result)})"

    def test_is_leaf_with_required_files(self):
        """Test leaf check with required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            np.save(tmpdir / "data.npy", np.array([1, 2, 3]))

            # Has required file
            is_leaf = DirectoryValidator.is_leaf_directory(
                tmpdir, required_files=["*.npy"]
            )
            assert (
                is_leaf is True
            ), f"Expected directory with .npy file to be leaf, got result: {is_leaf} (type: {type(is_leaf)})"

            # Missing required file
            is_leaf = DirectoryValidator.is_leaf_directory(
                tmpdir, required_files=["*.json"]
            )
            assert (
                is_leaf is False
            ), f"Expected directory without .json file to not be leaf, got result: {is_leaf} (type: {type(is_leaf)})"

    def test_validate_hierarchy_statistics(self):
        """Test hierarchy validation statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create 3-level hierarchy
            (tmpdir / "user1" / "screen1" / "camera1").mkdir(parents=True)
            (tmpdir / "user1" / "screen2" / "camera1").mkdir(parents=True)
            (tmpdir / "user2" / "screen1" / "camera1").mkdir(parents=True)

            stats = DirectoryValidator.validate_hierarchy(tmpdir)

            assert "depth_0" in stats, "Expected 'depth_0' key in stats (root level)"
            assert (
                stats["depth_1"] == 2
            ), f"Expected 2 users at depth_1, got {stats['depth_1']}"
            assert (
                stats["depth_2"] == 3
            ), f"Expected 3 screens at depth_2, got {stats['depth_2']}"
            assert (
                stats["depth_3"] == 3
            ), f"Expected 3 cameras at depth_3, got {stats['depth_3']}"
