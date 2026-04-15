"""Tests for validate_dataset_structure() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_dataset_structure


class TestValidateDatasetStructure:
    """Test validate_dataset_structure() function."""

    def test_valid_structure_with_default_keys(self):
        """Test dataset with required 'time' key."""
        dataset = {"time": np.array([0.0, 0.1, 0.2]), "data": np.array([1, 2, 3])}
        # Should not raise
        validate_dataset_structure(dataset)

    def test_valid_structure_with_custom_required_keys(self):
        """Test dataset with custom required keys."""
        dataset = {
            "time": np.array([0.0, 0.1]),
            "landmarks": np.array([[0, 0], [1, 1]]),
        }
        # Should not raise
        validate_dataset_structure(dataset, required_keys=["time", "landmarks"])

    def test_missing_required_key_raises(self):
        """Test that missing required key raises ValueError."""
        dataset = {"data": np.array([1, 2, 3])}

        with pytest.raises(ValueError, match="missing required keys"):
            validate_dataset_structure(dataset)

    def test_length_mismatch_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        dataset = {
            "time": np.array([0.0, 0.1, 0.2]),
            "data": np.array([1, 2]),  # Different length
        }

        with pytest.raises(ValueError, match="length mismatch"):
            validate_dataset_structure(dataset)

    def test_non_array_fields_ignored(self):
        """Test that non-array fields don't affect validation."""
        dataset = {
            "time": np.array([0.0, 0.1]),
            "data": np.array([1, 2]),
            "metadata": "string",  # Non-array
            "count": 42,  # Non-array
        }
        # Should not raise
        validate_dataset_structure(dataset)
