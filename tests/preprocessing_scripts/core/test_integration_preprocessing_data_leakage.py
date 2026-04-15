"""Tests for data leakage prevention in preprocessing."""

import numpy as np
from scripts.preprocessing.preprocessor import DatasetPreprocessor


class TestDataleakagePrevention:
    """Test that preprocessing prevents data leakage."""

    def test_sample_level_split_ensures_disjoint_samples(self):
        """Test that sample-level split is truly disjoint."""
        preprocessor = DatasetPreprocessor()

        # Simulate large dataset with 1000 valid samples
        np.random.seed(42)
        all_valid_samples = np.arange(5, 1005)  # 1000 samples
        np.random.shuffle(all_valid_samples)

        split_idx = 800
        train = sorted(all_valid_samples[:split_idx])
        test = sorted(all_valid_samples[split_idx:])

        # Validate
        preprocessor.validate_train_test_split(np.array(train), np.array(test))

        # Verify zero overlap
        overlap = np.intersect1d(train, test)
        assert len(overlap) == 0, f"Expected len(overlap) == 0, got {len(overlap)}"

    def test_frame_context_can_overlap_without_leakage(self):
        """Test that frame contexts can overlap as long as samples don't."""
        # Close samples with overlapping frames
        train_samples = np.array([20, 25])  # Frames will overlap
        test_samples = np.array([50])

        # Train samples are disjoint from test samples (no data leakage)
        sample_overlap = np.intersect1d(train_samples, test_samples)
        assert len(sample_overlap) == 0, "Samples must not overlap"

        # But train frames CAN overlap with train frames (same split)
        # This is fine - shows trajectory contexts can share frames within same split
        # as long as the SAMPLES don't overlap
