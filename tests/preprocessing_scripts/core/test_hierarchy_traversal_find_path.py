"""Tests for HierarchyTraversal find_path method."""

from pathlib import Path
import tempfile
from scripts.preprocessing.traversal import HierarchyTraversal


class TestHierarchyTraversalFindPath:
    """Test find_path method."""

    def test_find_existing_path(self):
        """Test finding existing path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "user1" / "screen1" / "camera1").mkdir(parents=True)

            traversal = HierarchyTraversal(tmpdir)
            found = traversal.find_path(
                {"userId": "user1", "screenId": "screen1", "cameraId": "camera1"}
            )

            assert found is not None, "Expected to find existing path, got None"
            assert (
                found.name == "camera1"
            ), f"Expected path name 'camera1', got '{found.name}'"

    def test_find_nonexistent_path(self):
        """Test finding nonexistent path returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "user1").mkdir()

            traversal = HierarchyTraversal(tmpdir)
            found = traversal.find_path({"userId": "user1", "screenId": "nonexistent"})

            assert found is None, f"Expected None for nonexistent path, got {found}"

    def test_find_partial_path(self):
        """Test finding partial path (only userId)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            user_path = tmpdir / "user1"
            user_path.mkdir()

            traversal = HierarchyTraversal(tmpdir)
            found = traversal.find_path({"userId": "user1"})

            assert found == user_path, f"Expected {user_path}, got {found}"
