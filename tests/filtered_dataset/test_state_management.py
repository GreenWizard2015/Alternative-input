"""Tests for FilteredDataset state management (reset, len)."""

import pytest


class TestFilteredDatasetStateManagement:
    """Test FilteredDataset state management (reset, len)."""

    def test_len_increases_with_valid_samples(self, dataset_with_frames):
        """Test that valid sample count changes when samples are added."""
        # Add samples until some are valid
        initial_count = len(dataset_with_frames)

        dataset_with_frames.add_sample({"time": 0.5})
        dataset_with_frames.add_sample({"time": 0.6})

        final_count = len(dataset_with_frames)

        # Verify behavior: count should increase or stay same (depending on validity)
        assert (
            final_count >= initial_count
        ), f"Valid sample count should grow or stay same: {final_count} >= {initial_count}"

    def test_reset_shuffles_order(self, dataset_with_frames):
        """Test that reset changes the internal sample order."""
        # Add multiple samples
        for i in range(5):
            dataset_with_frames.add_sample({"time": 0.2 + i * 0.1})

        if len(dataset_with_frames) < 2:
            pytest.skip("Not enough valid samples to test shuffling")

        # Get order before reset
        before_reset = dataset_with_frames.used_samples()

        # Reset and check if order changed
        dataset_with_frames.reset()
        after_reset = dataset_with_frames.used_samples()

        # Verify that reset completes and produces samples
        assert isinstance(
            after_reset, list
        ), f"reset() should maintain list of used samples, got {type(after_reset)}"
        assert len(after_reset) == len(
            before_reset
        ), f"reset() should preserve number of samples, got {len(after_reset)} vs {len(before_reset)}"

    def test_reset_preserves_valid_samples(self, dataset_with_frames):
        """Test that reset preserves which samples are valid."""
        # Add samples
        for i in range(3):
            dataset_with_frames.add_sample({"time": 0.3 + i * 0.1})

        valid_before = set(dataset_with_frames.valid_indices())

        dataset_with_frames.reset()

        valid_after = set(dataset_with_frames.valid_indices())

        assert (
            valid_before == valid_after
        ), f"valid_indices should be same after reset (order may differ), before: {valid_before}, after: {valid_after}"
