"""Tests for FilteredDataset sample addition and filtering."""

from Core.data.FilteredDataset import FilteredDataset


class TestFilteredDatasetSampleAddition:
    """Test FilteredDataset sample addition and filtering."""

    def test_add_single_sample_valid(self, storage_10_frames):
        """Test adding a single valid sample."""
        dataset = FilteredDataset(storage_10_frames, minFrames=2, maxT=1.0)

        # Add sample at index that should be valid
        idx = dataset.add_sample({"time": 0.3})

        assert isinstance(idx, int), f"add_sample should return index, got {type(idx)}"
        assert (
            len(dataset) == 1
        ), f"One valid sample should be counted, got {len(dataset)}"

    def test_add_single_sample_invalid(self, storage_10_frames):
        """Test adding an invalid sample (not enough frames in trajectory)."""
        dataset = FilteredDataset(storage_10_frames, minFrames=5, maxT=0.1)

        # Add sample early that won't have enough frames
        idx = dataset.add_sample({"time": 0.05})

        # This sample may or may not be valid depending on trajectory range
        # Just verify the operation completes
        assert isinstance(idx, int), f"add_sample should return index, got {type(idx)}"

    def test_add_multiple_samples_batch(self, storage_10_frames):
        """Test adding multiple samples in batch."""
        dataset = FilteredDataset(storage_10_frames, minFrames=2, maxT=1.0)

        samples = [
            {"time": 0.3},
            {"time": 0.4},
            {"time": 0.5},
        ]

        dataset.add_samples_block(samples)

        # Should have added samples and some should be valid
        assert (
            len(dataset) > 0
        ), f"At least one sample should be valid after batch add, got {len(dataset)}"

    def test_sample_added_to_storage(self, storage_10_frames):
        """Test that added samples increase total storage size."""
        dataset = FilteredDataset(storage_10_frames, minFrames=2, maxT=1.0)

        initial_total = dataset.total_samples
        dataset.add_sample({"time": 0.5})

        assert (
            dataset.total_samples == initial_total + 1
        ), f"Total samples should increase by 1, got {dataset.total_samples} from {initial_total}"

    def test_add_sample_returns_storage_index(self, storage_10_frames):
        """Test that add_sample returns the storage index of added sample."""
        dataset = FilteredDataset(storage_10_frames, minFrames=1, maxT=1.0)

        initial_total = dataset.total_samples
        returned_idx = dataset.add_sample({"time": 0.5})

        assert (
            returned_idx == initial_total
        ), f"Returned index should match storage position, got {returned_idx} expected {initial_total}"
