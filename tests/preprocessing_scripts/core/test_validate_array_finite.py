"""Tests for validate_array_finite() function."""

import pytest
import numpy as np
from scripts.preprocessing.validation import validate_array_finite


class TestValidateArrayFinite:
    """Test validate_array_finite() function."""

    def test_all_finite_values(self):
        """Test array with finite values."""
        arr = np.array([1.0, 2.5, 3.0, 4.5])
        validate_array_finite(arr)

    def test_nan_values_raise(self):
        """Test that NaN values raise ValueError."""
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            validate_array_finite(arr)

    def test_inf_values_raise(self):
        """Test that Inf values raise ValueError."""
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="Inf"):
            validate_array_finite(arr)

    def test_negative_inf_values_raise(self):
        """Test that negative Inf values raise ValueError."""
        arr = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            validate_array_finite(arr)

    def test_integer_array_skips_check(self):
        """Test that integer arrays skip finite check."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        validate_array_finite(arr)
