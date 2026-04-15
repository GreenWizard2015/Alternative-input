"""Tests for error messages in process_dataset."""

import pytest
import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestProcessDatasetErrorMessages:
    """Test that process_dataset() provides clear error messages."""

    def test_sparse_error_includes_delta_info(self):
        """Test that sparse dataset error includes delta information."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=0.1)

        dataset = {
            "time": np.array([0.0, 1.0, 2.0]),  # 1.0s gaps
            "data": np.array([1, 2, 3]),
        }

        with pytest.raises(ValueError) as exc_info:
            preprocessor.process_dataset(dataset)

        error_msg = str(exc_info.value)
        assert "sparse" in error_msg.lower(), "Error message should mention sparsity"

    def test_structure_error_includes_details(self):
        """Test that structure error includes helpful details."""
        preprocessor = DatasetPreprocessor()

        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([1, 2]),  # Wrong length
        }

        with pytest.raises(ValueError) as exc_info:
            preprocessor.process_dataset(dataset)

        error_msg = str(exc_info.value)
        assert "length" in error_msg.lower(), "Error message should mention length"
