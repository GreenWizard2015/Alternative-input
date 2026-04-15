"""Tests for validate_indices_are_sorted() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_indices_are_sorted


class TestValidateIndicesSorted:
    """Test validate_indices_are_sorted() function."""

    def test_strictly_sorted_indices(self):
        """Test strictly ascending indices."""
        indices = np.array([1, 5, 10, 20, 30])
        # Should not raise
        validate_indices_are_sorted(indices, allow_duplicates=False)

    def test_non_strictly_sorted_with_duplicates_allowed(self):
        """Test non-decreasing indices with duplicates allowed."""
        indices = np.array([1, 5, 5, 10, 20])
        # Should not raise
        validate_indices_are_sorted(indices, allow_duplicates=True)

    def test_duplicate_not_allowed_raises(self):
        """Test that duplicates raise when allow_duplicates=False."""
        indices = np.array([1, 5, 5, 10, 20])

        with pytest.raises(ValueError, match="not properly sorted"):
            validate_indices_are_sorted(indices, allow_duplicates=False)

    def test_unsorted_raises(self):
        """Test that unsorted indices raise ValueError."""
        indices = np.array([1, 10, 5, 20])

        with pytest.raises(ValueError, match="not properly sorted"):
            validate_indices_are_sorted(indices)

    def test_single_index_trivially_sorted(self):
        """Test that single index is trivially sorted."""
        indices = np.array([5])
        # Should not raise
        validate_indices_are_sorted(indices)

    def test_empty_indices_trivially_sorted(self):
        """Test that empty indices are trivially sorted."""
        indices = np.array([], dtype=np.int64)
        # Should not raise
        validate_indices_are_sorted(indices)
