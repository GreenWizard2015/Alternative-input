"""Tests for validate_array_dtype_compatible() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_array_dtype_compatible


class TestValidateArrayDtype:
    """Test validate_array_dtype_compatible() function."""

    def test_integer_dtype_numeric(self):
        """Test integer array matches numeric family."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        # Should not raise
        validate_array_dtype_compatible(arr, "numeric")

    def test_float_dtype_numeric(self):
        """Test float array matches numeric family."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Should not raise
        validate_array_dtype_compatible(arr, "numeric")

    def test_integer_dtype_integer(self):
        """Test integer array matches integer family."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        # Should not raise
        validate_array_dtype_compatible(arr, "integer")

    def test_float_dtype_float(self):
        """Test float array matches float family."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        # Should not raise
        validate_array_dtype_compatible(arr, "float")

    def test_wrong_dtype_raises(self):
        """Test that wrong dtype raises ValueError."""
        arr = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="expected integer"):
            validate_array_dtype_compatible(arr, "integer")
