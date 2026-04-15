"""Tests for validate_train_test_split method."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestValidateTrainTestSplit:
    """Test validate_train_test_split() method."""

    def test_disjoint_split_valid(self):
        """Test that disjoint train/test split is valid."""
        preprocessor = DatasetPreprocessor()

        train_indices = np.array([0, 1, 2, 3, 4])
        test_indices = np.array([5, 6, 7, 8, 9])

        # Should not raise
        preprocessor.validate_train_test_split(train_indices, test_indices)

    def test_overlapping_split_raises(self):
        """Test that overlapping split raises."""
        preprocessor = DatasetPreprocessor()

        train_indices = np.array([0, 1, 2, 5, 6])
        test_indices = np.array([3, 4, 5, 7, 8])  # 5 is in both

        with pytest.raises(ValueError, match="overlap"):
            preprocessor.validate_train_test_split(train_indices, test_indices)

    def test_completely_overlapping_split_raises(self):
        """Test that completely overlapping split raises."""
        preprocessor = DatasetPreprocessor()

        indices = np.array([0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="overlap"):
            preprocessor.validate_train_test_split(indices, indices)

    def test_empty_train_valid(self):
        """Test that empty train set is valid (edge case)."""
        preprocessor = DatasetPreprocessor()

        train_indices = np.array([], dtype=np.int64)
        test_indices = np.array([0, 1, 2, 3])

        # Should not raise
        preprocessor.validate_train_test_split(train_indices, test_indices)

    def test_empty_test_valid(self):
        """Test that empty test set is valid (edge case)."""
        preprocessor = DatasetPreprocessor()

        train_indices = np.array([0, 1, 2, 3])
        test_indices = np.array([], dtype=np.int64)

        # Should not raise
        preprocessor.validate_train_test_split(train_indices, test_indices)

    def test_both_empty_valid(self):
        """Test that both empty is valid (trivial case)."""
        preprocessor = DatasetPreprocessor()

        train_indices = np.array([], dtype=np.int64)
        test_indices = np.array([], dtype=np.int64)

        # Should not raise
        preprocessor.validate_train_test_split(train_indices, test_indices)
