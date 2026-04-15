"""Tests for HierarchyTraversal initialization."""

import pytest
import tempfile
from pathlib import Path
from scripts.preprocessing.traversal import HierarchyTraversal


class TestHierarchyTraversalInitialization:
    """Test HierarchyTraversal initialization."""

    def test_valid_initialization(self):
        """Test initialization with valid root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(Path(tmpdir))
            assert hasattr(
                traversal, "root_dir"
            ), "Traversal should have root_dir attribute"
            assert traversal.root_dir == Path(
                tmpdir
            ), f"Expected root_dir to be {Path(tmpdir)}, got {traversal.root_dir}"

    def test_nonexistent_directory_raises(self):
        """Test that nonexistent directory raises ValueError."""
        nonexistent = Path("/nonexistent/path/to/directory")
        with pytest.raises(ValueError, match="does not exist"):
            HierarchyTraversal(nonexistent)

    def test_file_instead_of_directory_raises(self):
        """Test that file path raises ValueError."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="not a directory"):
                HierarchyTraversal(Path(tmpfile.name))

    def test_string_path_converted_to_path(self):
        """Test that string paths are handled (converted to Path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = HierarchyTraversal(tmpdir)  # Pass string
            assert isinstance(
                traversal.root_dir, Path
            ), f"Expected Path object, got {type(traversal.root_dir)}: {traversal.root_dir}"
