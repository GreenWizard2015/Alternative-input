"""Tests for error handling in preprocessing pipeline."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from scripts.preprocessing.preprocessor import DatasetPreprocessor
from scripts.preprocessing.traversal import DirectoryValidator


class TestErrorHandlingInPipeline:
    """Test error handling in preprocessing pipeline."""

    def test_invalid_dataset_structure_caught_early(self):
        """Test that invalid datasets are caught early."""
        preprocessor = DatasetPreprocessor()

        invalid_dataset = {
            "data": np.array([1, 2, 3])
            # Missing required "time" key
        }

        with pytest.raises(ValueError, match="time"):
            preprocessor.process_dataset(invalid_dataset)

    def test_sparse_data_rejected(self):
        """Test that sparse datasets are rejected."""
        preprocessor = DatasetPreprocessor(max_delta_threshold=0.1)

        sparse_dataset = {
            "time": np.array([0.0, 1.0, 2.0, 3.0]),  # 1.0s gaps
            "data": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(ValueError, match="sparse"):
            preprocessor.process_dataset(sparse_dataset)

    def test_trajectory_boundary_violation_caught(self):
        """Test that samples too close to boundary are caught."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=10)

        sample_indices = np.array([5])  # Too close to start

        with pytest.raises(ValueError, match="minimum"):
            preprocessor.expand_sample_indices(sample_indices, dataset_size=50)

    def test_missing_required_files_caught(self):
        """Test that missing required files are caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Folder exists but no files
            validator = DirectoryValidator()
            is_valid = validator.is_leaf_directory(tmpdir, required_files=["*.npy"])
            assert is_valid is False, "Expected is_valid to be False"
