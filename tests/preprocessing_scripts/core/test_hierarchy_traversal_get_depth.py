"""Tests for HierarchyTraversal get_depth method."""

from pathlib import Path
import tempfile
from scripts.preprocessing.traversal import HierarchyTraversal


class TestHierarchyTraversalGetDepth:
    """Test get_depth method."""

    def test_root_depth_is_zero(self):
        """Test that root has depth 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))
            depth = traversal.get_depth(Path(tmpdir))

            assert depth == 0, f"Expected root depth to be 0, got {depth}"

    def test_depth_calculation(self):
        """Test depth calculation for nested paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "user1" / "screen1" / "camera1").mkdir(parents=True)

            traversal = HierarchyTraversal(tmpdir)

            depth1 = traversal.get_depth(tmpdir / "user1")
            depth2 = traversal.get_depth(tmpdir / "user1" / "screen1")
            depth3 = traversal.get_depth(tmpdir / "user1" / "screen1" / "camera1")

            assert depth1 == 1, f"Expected depth 1 for user1, got {depth1}"
            assert depth2 == 2, f"Expected depth 2 for screen1, got {depth2}"
            assert depth3 == 3, f"Expected depth 3 for camera1, got {depth3}"

    def test_depth_outside_root_returns_minus_one(self):
        """Test that paths outside root return -1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))
            depth = traversal.get_depth(Path("/some/other/path"))

            assert depth == -1, f"Expected -1 for path outside root, got {depth}"
