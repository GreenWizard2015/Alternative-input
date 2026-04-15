"""Tests for validate_index_subset() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_index_subset


class TestValidateIndexSubset:
    """Test validate_index_subset() function."""

    def test_valid_subset(self):
        """Test that valid subset passes."""
        subset = np.array([1, 5, 20])
        superset = np.array([1, 5, 10, 20, 30])
        # Should not raise
        validate_index_subset(subset, superset)

    def test_subset_not_in_superset_raises(self):
        """Test that elements not in superset raise ValueError."""
        subset = np.array([1, 5, 50])
        superset = np.array([1, 5, 10, 20, 30])

        with pytest.raises(ValueError, match="not in"):
            validate_index_subset(subset, superset)

    def test_equal_sets_valid_subset(self):
        """Test that equal sets are valid subsets."""
        indices = np.array([1, 5, 10])
        # Should not raise
        validate_index_subset(indices, indices)

    def test_empty_subset_valid(self):
        """Test that empty subset is valid."""
        subset = np.array([], dtype=np.int64)
        superset = np.array([1, 5, 10])
        # Should not raise
        validate_index_subset(subset, superset)
