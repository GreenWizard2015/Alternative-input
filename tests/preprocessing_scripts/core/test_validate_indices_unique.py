"""Tests for validate_indices_unique() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_indices_unique


class TestValidateIndicesUnique:
    """Test validate_indices_unique() function."""

    def test_all_unique_indices(self):
        """Test that all unique indices pass."""
        indices = np.array([1, 5, 10, 20, 30])
        # Should not raise
        validate_indices_unique(indices)

    def test_duplicate_indices_raise(self):
        """Test that duplicates raise ValueError."""
        indices = np.array([1, 5, 10, 10, 20])

        with pytest.raises(ValueError, match="duplicate"):
            validate_indices_unique(indices)

    def test_empty_indices_unique(self):
        """Test that empty indices are trivially unique."""
        indices = np.array([], dtype=np.int64)
        # Should not raise
        validate_indices_unique(indices)
