"""Tests for dataset_loading.dataset_from_stats() sorting fix."""

import tempfile
import os
import json
from Core.dataset_loading import dataset_from_stats
from Core.Constants import ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE
from Core.utils.DatasetPath import DatasetPath


class TestDatasetFromStatsSorting:
    """Test dataset_from_stats ID list sorting for deterministic indices."""

    def _create_mock_folder_structure(
        self, tmpdir, user_ids, screen_ids, camera_ids, monitor_ids, place_ids
    ):
        """Create mock dataset folder structure for testing.

        Creates nested directories for each ID combination to allow dataset_from_stats
        to find valid dataset folders.
        """
        for user_id in user_ids:
            for screen_id in screen_ids:
                for camera_id in camera_ids:
                    for monitor_id in monitor_ids:
                        for place_id in place_ids:
                            path = DatasetPath(
                                user_id,
                                screen_id,
                                camera_id,
                                monitor_id,
                                place_id,
                                base_path=tmpdir,
                            )
                            os.makedirs(path.full_path, exist_ok=True)

    def test_unsorted_ids_are_sorted_before_enumeration(self):
        """Test that unsorted ID lists are sorted before enumeration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Unsorted IDs
            user_ids = ["uuid_z", "uuid_a", "uuid_m"]
            screen_ids = ["screen_b", "screen_a"]
            camera_ids = ["cam_2", "cam_1", "cam_0"]
            monitor_ids = ["mon_b", "mon_a"]
            place_ids = ["place_z", "place_a"]

            # Create folder structure
            self._create_mock_folder_structure(
                tmpdir, user_ids, screen_ids, camera_ids, monitor_ids, place_ids
            )

            # Create stats with unsorted lists
            stats = {
                ID_USER: user_ids,
                ID_SCREEN: screen_ids,
                ID_CAMERA: camera_ids,
                ID_MONITOR: monitor_ids,
                ID_PLACE: place_ids,
                "blacklist": [],
            }

            # Write stats to JSON file
            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(stats, f)

            # Collect all results
            results = list(dataset_from_stats(json_path))
            assert len(results) > 0, "Should find at least one dataset"

            # Check that indices are deterministic
            first_result = results[0]

            # First result should have index 0 for all IDs (after sorting)
            # because sorted lists will be: [uuid_a, uuid_m, uuid_z], [screen_a, screen_b], etc.
            assert (
                first_result.user_idx == 0
            ), "uuid_a should be at index 0 (first alphabetically)"
            assert (
                first_result.screen_idx == 0
            ), "screen_a should be at index 0 (first alphabetically)"

    def test_same_dataset_gets_same_index_regardless_of_stats_order(self):
        """Test that same dataset gets same indices even if stats order changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup IDs (order will vary in stats)
            all_user_ids = ["user_x", "user_y", "user_z"]
            all_screen_ids = ["screen_1", "screen_2"]
            all_camera_ids = ["camera_a", "camera_b"]
            all_monitor_ids = ["monitor_alpha"]
            all_place_ids = ["place_home"]

            # Create folder structure
            self._create_mock_folder_structure(
                tmpdir,
                all_user_ids,
                all_screen_ids,
                all_camera_ids,
                all_monitor_ids,
                all_place_ids,
            )

            # First stats with one order
            stats1 = {
                ID_USER: ["user_z", "user_x", "user_y"],  # Unsorted
                ID_SCREEN: ["screen_2", "screen_1"],  # Unsorted
                ID_CAMERA: ["camera_b", "camera_a"],  # Unsorted
                ID_MONITOR: ["monitor_alpha"],
                ID_PLACE: ["place_home"],
                "blacklist": [],
            }

            # Second stats with different order
            stats2 = {
                ID_USER: ["user_y", "user_z", "user_x"],  # Different order
                ID_SCREEN: ["screen_1", "screen_2"],  # Different order
                ID_CAMERA: ["camera_a", "camera_b"],  # Different order
                ID_MONITOR: ["monitor_alpha"],
                ID_PLACE: ["place_home"],
                "blacklist": [],
            }

            # Write stats to JSON files
            json_path1 = os.path.join(tmpdir, "stats1.json")
            json_path2 = os.path.join(tmpdir, "stats2.json")
            with open(json_path1, "w") as f:
                json.dump(stats1, f)
            with open(json_path2, "w") as f:
                json.dump(stats2, f)

            # Get results from both
            results1 = list(dataset_from_stats(json_path1))
            results2 = list(dataset_from_stats(json_path2))

            assert len(results1) == len(
                results2
            ), f"Both should find same number of datasets, got {len(results1)} vs {len(results2)}"

            # For a specific dataset (user_x, screen_1, camera_a, monitor_alpha, place_home)
            # find it in both result sets and verify indices are the same
            r1 = next(
                (
                    r
                    for r in results1
                    if r.path.user_id == "user_x" and r.path.screen_id == "screen_1"
                ),
                None,
            )
            r2 = next(
                (
                    r
                    for r in results2
                    if r.path.user_id == "user_x" and r.path.screen_id == "screen_1"
                ),
                None,
            )

            assert r1 is not None, "Dataset (user_x, screen_1) should exist in results1"
            assert r2 is not None, "Dataset (user_x, screen_1) should exist in results2"

            # Indices should be identical
            assert (
                r1.user_idx == r2.user_idx
            ), f"user_idx mismatch: {r1.user_idx} vs {r2.user_idx}"
            assert (
                r1.screen_idx == r2.screen_idx
            ), f"screen_idx mismatch: {r1.screen_idx} vs {r2.screen_idx}"
            assert (
                r1.camera_idx == r2.camera_idx
            ), f"camera_idx mismatch: {r1.camera_idx} vs {r2.camera_idx}"

    def test_deterministic_indices_match_sorted_order(self):
        """Test that indices correspond to sorted order of IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_ids = ["alpha", "gamma", "beta"]
            screen_ids = ["z_screen", "a_screen"]

            # Create minimal folder structure
            for user_id in user_ids:
                for screen_id in screen_ids:
                    path = DatasetPath(
                        user_id, screen_id, "cam0", "mon0", "place0", base_path=tmpdir
                    )
                    os.makedirs(path.full_path, exist_ok=True)

            stats = {
                ID_USER: user_ids,  # ["alpha", "gamma", "beta"]
                ID_SCREEN: screen_ids,  # ["z_screen", "a_screen"]
                ID_CAMERA: ["cam0"],
                ID_MONITOR: ["mon0"],
                ID_PLACE: ["place0"],
                "blacklist": [],
            }

            # Write stats to JSON file
            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(stats, f)

            results = list(dataset_from_stats(json_path))

            # After sorting: users become ["alpha", "beta", "gamma"]
            # After sorting: screens become ["a_screen", "z_screen"]

            # Find indices for specific datasets
            alpha_a = next(
                (
                    r
                    for r in results
                    if r.path.user_id == "alpha" and r.path.screen_id == "a_screen"
                ),
                None,
            )
            beta_z = next(
                (
                    r
                    for r in results
                    if r.path.user_id == "beta" and r.path.screen_id == "z_screen"
                ),
                None,
            )
            gamma_a = next(
                (
                    r
                    for r in results
                    if r.path.user_id == "gamma" and r.path.screen_id == "a_screen"
                ),
                None,
            )

            # Verify indices match alphabetical order
            assert (
                alpha_a.user_idx == 0
            ), f"alpha should be index 0, got {alpha_a.user_idx}"
            assert (
                beta_z.user_idx == 1
            ), f"beta should be index 1, got {beta_z.user_idx}"
            assert (
                gamma_a.user_idx == 2
            ), f"gamma should be index 2, got {gamma_a.user_idx}"

            assert (
                alpha_a.screen_idx == 0
            ), f"a_screen should be index 0, got {alpha_a.screen_idx}"
            assert (
                beta_z.screen_idx == 1
            ), f"z_screen should be index 1, got {beta_z.screen_idx}"

    def test_blacklist_works_with_sorted_ids(self):
        """Test that blacklist filtering works correctly with sorted IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_ids = ["user_1", "user_2", "user_3"]
            screen_ids = ["screen_a", "screen_b"]

            # Create folder structure
            for user_id in user_ids:
                for screen_id in screen_ids:
                    path = DatasetPath(
                        user_id, screen_id, "cam", "mon", "place", base_path=tmpdir
                    )
                    os.makedirs(path.full_path, exist_ok=True)

            # Blacklist one specific combination (user_2, screen_a)
            stats = {
                ID_USER: ["user_3", "user_1", "user_2"],  # Unsorted
                ID_SCREEN: ["screen_b", "screen_a"],  # Unsorted
                ID_CAMERA: ["cam"],
                ID_MONITOR: ["mon"],
                ID_PLACE: ["place"],
                "blacklist": [["user_2", "screen_a", "cam", "mon", "place"]],
            }

            # Write stats to JSON file
            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(stats, f)

            results = list(dataset_from_stats(json_path))

            # Should have 5 results (3 users * 2 screens - 1 blacklisted)
            assert (
                len(results) == 5
            ), f"Expected 5 results (3 users * 2 screens - 1), got {len(results)}"

            # Verify blacklisted combination is not in results
            blacklisted = any(
                r.path.user_id == "user_2" and r.path.screen_id == "screen_a"
                for r in results
            )
            assert (
                not blacklisted
            ), "Blacklisted dataset (user_2, screen_a) should not be in results"

            # Verify other combinations are present
            user2_screenb = any(
                r.path.user_id == "user_2" and r.path.screen_id == "screen_b"
                for r in results
            )
            assert (
                user2_screenb
            ), "Non-blacklisted user_2, screen_b combination should exist in results"

    def test_empty_stats_yields_nothing(self):
        """Test that empty stats (no matching folders) yields nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = {
                ID_USER: ["nonexistent"],
                ID_SCREEN: ["nonexistent"],
                ID_CAMERA: ["nonexistent"],
                ID_MONITOR: ["nonexistent"],
                ID_PLACE: ["nonexistent"],
                "blacklist": [],
            }

            # Write stats to JSON file
            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(stats, f)

            results = list(dataset_from_stats(json_path))
            assert (
                len(results) == 0
            ), f"Should yield nothing for nonexistent folders, got {len(results)} results"

    def test_single_element_lists_work(self):
        """Test that single-element ID lists work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create single folder
            path = DatasetPath(
                "user", "screen", "cam", "mon", "place", base_path=tmpdir
            )
            os.makedirs(path.full_path, exist_ok=True)

            stats = {
                ID_USER: ["user"],
                ID_SCREEN: ["screen"],
                ID_CAMERA: ["cam"],
                ID_MONITOR: ["mon"],
                ID_PLACE: ["place"],
                "blacklist": [],
            }

            # Write stats to JSON file
            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(stats, f)

            results = list(dataset_from_stats(json_path))
            assert (
                len(results) == 1
            ), f"Should have exactly one result for single-element lists, got {len(results)}"
            assert (
                results[0].user_idx == 0
            ), f"Single user should have index 0, got {results[0].user_idx}"
            assert (
                results[0].screen_idx == 0
            ), f"Single screen should have index 0, got {results[0].screen_idx}"
