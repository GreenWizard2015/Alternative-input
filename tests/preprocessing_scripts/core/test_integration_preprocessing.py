"""Integration tests for complete preprocessing workflow.

Tests the full pipeline: load → preprocess → expand → split → validate
"""

import numpy as np
from pathlib import Path
import tempfile
from scripts.preprocessing.preprocessor import DatasetPreprocessor
from scripts.preprocessing.traversal import (
    HierarchyTraversal,
    DatasetLoader,
    DirectoryValidator,
)


class TestFullPreprocessingPipeline:
    """Test complete preprocessing workflow end-to-end."""

    def test_load_and_preprocess_single_dataset(self):
        """Test loading and preprocessing a single dataset file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dataset
            dataset = {
                "time": np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4]),
                "data": np.array([1, 2, 3, 4, 5, 6]),
            }

            # Save dataset
            np.save(tmpdir / "time.npy", dataset["time"])
            np.save(tmpdir / "data.npy", dataset["data"])

            # Load dataset
            loader = DatasetLoader(tmpdir)
            loaded = loader.load_dataset(tmpdir)

            assert loaded is not None, f"loaded should not be None, got: {loaded}"
            assert (
                "time" in loaded
            ), f"Result should contain 'time' key, got keys: {list(loaded.keys())}"
            assert (
                len(loaded["time"]) == 6
            ), f"Expected 6 time values, got {len(loaded['time'])} values: {loaded['time']}"

            # Preprocess
            preprocessor = DatasetPreprocessor(
                min_trajectory_frames=5, max_delta_threshold=0.3
            )
            processed = preprocessor.process_dataset(loaded)

            # Should remove one zero-delta frame
            assert len(processed["time"]) == 5, "One duplicate should be removed"

    def test_traverse_and_preprocess_hierarchy(self):
        """Test traversing hierarchy and preprocessing multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create hierarchy with datasets
            user1_screen1 = tmpdir / "user1" / "screen1"
            user1_screen1.mkdir(parents=True)
            user2_screen1 = tmpdir / "user2" / "screen1"
            user2_screen1.mkdir(parents=True)

            # Create datasets at each location
            for folder in [user1_screen1, user2_screen1]:
                time = np.linspace(0, 1, 20)
                data = np.arange(20) * 10
                np.save(folder / "time.npy", time)
                np.save(folder / "data.npy", data)

            # Traverse and preprocess
            traversal = HierarchyTraversal(tmpdir)
            loader = DatasetLoader(tmpdir)
            preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

            processed_count = 0
            processed = None
            for user_id, user_path in traversal.iterate_level("userId"):
                for screen_path in traversal.list_subdirectories(user_path):
                    if screen_path.is_dir():
                        dataset = loader.load_dataset(screen_path)
                        if dataset:
                            processed = preprocessor.process_dataset(
                                dataset, validate_only=True
                            )
                            processed_count += 1

            assert processed is not None, f"Processing should succeed, got: {processed}"
            assert (
                "time" in processed
            ), f"Result should contain 'time' key, got keys: {list(processed.keys())}"

            assert (
                processed_count >= 2
            ), f"Should process at least 2 datasets, got {processed_count} datasets"

    def test_extract_indices_and_validate_split(self):
        """Test extracting sample indices and validating train/test split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Setup preprocessor
            preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

            # Simulate sample indices (e.g., from FilteredDataset)
            all_valid_indices = np.arange(5, 100)  # Must be >= min_trajectory_frames
            np.random.seed(42)
            np.random.shuffle(all_valid_indices)

            split_point = int(0.8 * len(all_valid_indices))
            train_indices = sorted(all_valid_indices[:split_point])
            test_indices = sorted(all_valid_indices[split_point:])

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            # Validate split
            preprocessor.validate_train_test_split(train_indices, test_indices)

            # Verify completeness
            total = len(train_indices) + len(test_indices)
            assert total == len(all_valid_indices), "Total should match original count"

            # Verify disjointness
            assert (
                len(np.intersect1d(train_indices, test_indices)) == 0
            ), "Train/test should have no overlap"

    def test_expand_indices_for_both_splits(self):
        """Test expanding indices for both train and test splits."""
        with tempfile.TemporaryDirectory() as _:
            # Dataset with 100 frames
            dataset_size = 100
            preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

            # Sample indices for train and test
            train_samples = np.array([10, 20, 30, 40])
            test_samples = np.array([60, 70, 80, 90])

            # Expand both
            train_frames = preprocessor.expand_sample_indices(
                train_samples, dataset_size
            )
            test_frames = preprocessor.expand_sample_indices(test_samples, dataset_size)

            # Verify frames are disjoint (frame-level split can overlap, but sample-level must not)
            # Actually, frame-level can overlap for data augmentation purposes, but sample-level must be disjoint
            sample_overlap = np.intersect1d(train_samples, test_samples)
            assert len(sample_overlap) == 0, "Sample-level split must be disjoint"

            # Verify all samples are in their respective frame sets
            assert np.all(
                np.isin(train_samples, train_frames)
            ), "All train samples should be in train frames"
            assert np.all(
                np.isin(test_samples, test_frames)
            ), "All test samples should be in test frames"

    def test_compute_frame_mappings_for_trajectory_extraction(self):
        """Test computing frame mappings for trajectory context extraction."""
        preprocessor = DatasetPreprocessor(min_trajectory_frames=5)

        # Create sample indices
        samples = np.array([10, 20, 30])
        mappings = preprocessor.compute_sample_frame_mapping(samples)

        # Create dummy dataset
        dataset = {"time": np.linspace(0, 3, 40), "data": np.arange(40)}

        # Extract contexts for each sample
        contexts = {}
        for sample_idx in samples:
            start, end = mappings[sample_idx]
            context = {"data": dataset["data"][start:end]}
            contexts[sample_idx] = context

            # Verify context includes the sample
            assert (
                dataset["data"][sample_idx] in context["data"]
            ), "Context should include sample data"

        assert len(contexts) == len(samples), f"Should have {len(samples)} contexts"

    def test_directory_validation_before_preprocessing(self):
        """Test validating directory structure before preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create valid hierarchy
            (tmpdir / "user1" / "screen1").mkdir(parents=True)
            np.save(tmpdir / "user1" / "screen1" / "data.npy", np.array([1, 2, 3]))

            # Validate
            validator = DirectoryValidator()
            is_leaf = validator.is_leaf_directory(
                tmpdir / "user1" / "screen1", required_files=["*.npy"]
            )
            assert is_leaf is True, "Directory with .npy files should be a leaf"

            # Can now safely preprocess this leaf
            loader = DatasetLoader(tmpdir)
            dataset = loader.load_dataset(tmpdir / "user1" / "screen1")
            assert dataset is not None, "Should load dataset successfully"
