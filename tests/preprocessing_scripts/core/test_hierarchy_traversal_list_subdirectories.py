"""Tests for HierarchyTraversal list_subdirectories method."""

from pathlib import Path
import tempfile
from scripts.preprocessing.traversal import HierarchyTraversal


class TestHierarchyTraversalListSubdirectories:
    """Test list_subdirectories method."""

    def test_list_subdirectories(self):
        """Test listing subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create some subdirectories
            (tmpdir / "dir1").mkdir()
            (tmpdir / "dir2").mkdir()
            (tmpdir / "file.txt").touch()

            traversal = HierarchyTraversal(tmpdir)
            subdirs = traversal.list_subdirectories(tmpdir)

            assert len(subdirs) == 2, f"Expected 2 subdirectories, got {len(subdirs)}"
            assert all(
                p.is_dir() for p in subdirs
            ), "Expected all returned paths to be directories"

    def test_list_subdirectories_sorted(self):
        """Test that subdirectories are returned sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for name in ["z_dir", "a_dir", "m_dir"]:
                (tmpdir / name).mkdir()

            traversal = HierarchyTraversal(tmpdir)
            subdirs = traversal.list_subdirectories(tmpdir)
            names = [p.name for p in subdirs]

            assert names == [
                "a_dir",
                "m_dir",
                "z_dir",
            ], f"Expected sorted names ['a_dir', 'm_dir', 'z_dir'], got {names}"

    def test_list_empty_directory(self):
        """Test listing empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))
            subdirs = traversal.list_subdirectories(Path(tmpdir))

            assert (
                subdirs == []
            ), f"Expected empty list for empty directory, got {subdirs}"
