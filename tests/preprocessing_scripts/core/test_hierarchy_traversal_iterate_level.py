"""Tests for HierarchyTraversal iterate_level method."""

import pytest
from pathlib import Path
import tempfile
from scripts.preprocessing.traversal import HierarchyTraversal


class TestHierarchyTraversalIterateLevel:
    """Test iterate_level method."""

    def test_iterate_level_invalid_name_raises(self):
        """Test that invalid level name raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))

            with pytest.raises(ValueError, match="Unknown level"):
                list(traversal.iterate_level("invalidLevel"))

    def test_iterate_level_valid_names(self):
        """Test that valid level names are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))

            # Should not raise
            for level in ["userId", "screenId", "cameraId", "monitorId", "placeId"]:
                result = list(traversal.iterate_level(level))
                assert isinstance(
                    result, list
                ), f"Expected list for level '{level}', got {type(result)}"

    def test_iterate_level_empty_directory(self):
        """Test iterating level in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))
            result = list(traversal.iterate_level("userId"))

            assert (
                result == []
            ), f"Expected empty list for empty directory, got {result}"

    def test_iterate_level_with_hierarchy(self):
        """Test iterating with actual hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create hierarchy: userId/screenId/cameraId
            (tmpdir / "user1" / "screen1" / "camera1").mkdir(parents=True)
            (tmpdir / "user1" / "screen2" / "camera1").mkdir(parents=True)
            (tmpdir / "user2" / "screen1" / "camera1").mkdir(parents=True)

            traversal = HierarchyTraversal(tmpdir)

            # Iterate userId level (should be at root)
            users = list(traversal.iterate_level("userId"))
            assert len(users) == 2, f"Expected 2 users, got {len(users)}"
            user_ids = [uid for uid, _ in users]
            assert (
                "user1" in user_ids and "user2" in user_ids
            ), f"Expected user1 and user2 in {user_ids}"
