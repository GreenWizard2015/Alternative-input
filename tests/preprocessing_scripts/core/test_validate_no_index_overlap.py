"""Tests for validate_no_index_overlap() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_no_index_overlap


class TestValidateNoIndexOverlap:
    """Test validate_no_index_overlap() function."""

    def test_disjoint_indices(self):
        """Test that disjoint index sets pass."""
        indices1 = np.array([1, 5, 10])
        indices2 = np.array([20, 30, 40])
        # Should not raise
        validate_no_index_overlap(indices1, indices2)

    def test_overlapping_indices_raise(self):
        """Test that overlapping indices raise ValueError."""
        indices1 = np.array([1, 5, 10, 20])
        indices2 = np.array([15, 20, 25, 30])

        with pytest.raises(ValueError, match="overlap"):
            validate_no_index_overlap(indices1, indices2)

    def test_completely_overlapping_indices_raise(self):
        """Test complete overlap raises."""
        indices1 = np.array([1, 5, 10])
        indices2 = np.array([1, 5, 10])

        with pytest.raises(ValueError, match="overlap"):
            validate_no_index_overlap(indices1, indices2)
