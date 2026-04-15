"""Tests for FilterDataLoader and FilterTestLoader classes."""

import numpy as np
from pathlib import Path
from Core.data.FilterDataLoader import FilterDataLoader


class TestFilterDataLoader:
    """Test FilterDataLoader initialization and data loading."""

    def create_test_npz(self, tmp_path: Path, filename: str) -> Path:
        """Create a test NPZ file with filter data."""
        data = {
            "left eye": np.random.randint(0, 255, (10, 48, 48), dtype=np.uint8),
            "right eye": np.random.randint(0, 255, (10, 48, 48), dtype=np.uint8),
            "is_valid": np.random.randint(0, 2, (10,), dtype=np.int32),
        }

        file_path = tmp_path / filename
        np.savez(file_path, **data)
        return file_path

    def test_filter_data_loader_initialization(self, tmp_path: Path):
        """Test FilterDataLoader initializes with valid parameters."""
        # Create test data
        train_file = self.create_test_npz(tmp_path, "filter-train.npz")

        # Should initialize successfully
        loader = FilterDataLoader(str(train_file), batch_size=4)

        # Check attributes
        assert len(loader) == 3  # ceil(10 / 4) = 3 batches
        assert hasattr(loader, "_batch_size") and loader._batch_size == 4
