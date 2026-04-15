"""Tests for dataset_loading.ensure_sorted_stats() function."""

import tempfile
import os
import json
import pytest
from Core.dataset_loading import ensure_sorted_stats, read_json
from Core.Constants import ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE


class TestEnsureSortedStats:
    """Test ensure_sorted_stats() function for ID sorting."""

    def test_ensure_sorted_stats_sorts_all_lists(self):
        """Test that all ID lists are sorted."""
        stats = {
            ID_USER: ["user_z", "user_a", "user_m"],
            ID_SCREEN: ["screen_b", "screen_a"],
            ID_CAMERA: ["cam_2", "cam_1", "cam_0"],
            ID_MONITOR: ["mon_b", "mon_a"],
            ID_PLACE: ["place_z", "place_a"],
            "blacklist": [],
        }

        sorted_stats = ensure_sorted_stats(stats)

        assert sorted_stats[ID_USER] == [
            "user_a",
            "user_m",
            "user_z",
        ], f"Expected sorted user list, got {sorted_stats[ID_USER]}"
        assert sorted_stats[ID_SCREEN] == [
            "screen_a",
            "screen_b",
        ], f"Expected sorted screen list, got {sorted_stats[ID_SCREEN]}"
        assert sorted_stats[ID_CAMERA] == [
            "cam_0",
            "cam_1",
            "cam_2",
        ], f"Expected sorted camera list, got {sorted_stats[ID_CAMERA]}"
        assert sorted_stats[ID_MONITOR] == [
            "mon_a",
            "mon_b",
        ], f"Expected sorted monitor list, got {sorted_stats[ID_MONITOR]}"
        assert sorted_stats[ID_PLACE] == [
            "place_a",
            "place_z",
        ], f"Expected sorted place list, got {sorted_stats[ID_PLACE]}"

    def test_ensure_sorted_stats_preserves_values(self):
        """Test that all values are preserved (only order changes)."""
        stats = {
            ID_USER: ["z", "a"],
            ID_SCREEN: ["screen"],
            ID_CAMERA: ["cam"],
            ID_MONITOR: ["mon"],
            ID_PLACE: ["place"],
            "blacklist": [["a", "screen", "cam", "mon", "place"]],
            "extra_key": "extra_value",
        }

        sorted_stats = ensure_sorted_stats(stats)

        # Check all values are present
        assert set(sorted_stats[ID_USER]) == {
            "z",
            "a",
        }, f"Expected user values {{'z', 'a'}}, got {set(sorted_stats[ID_USER])}"
        assert sorted_stats["blacklist"] == [
            ["a", "screen", "cam", "mon", "place"]
        ], f"Expected blacklist to be preserved, got {sorted_stats['blacklist']}"
        assert (
            sorted_stats["extra_key"] == "extra_value"
        ), f"Expected extra_key='extra_value', got {sorted_stats.get('extra_key')}"

    def test_ensure_sorted_stats_raises_on_missing_keys(self):
        """Test that ValueError is raised if required keys missing."""
        stats = {
            ID_USER: ["user"],
            # Missing ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE
        }

        with pytest.raises(ValueError, match="stats missing required keys"):
            ensure_sorted_stats(stats)

    def test_read_json_sorts_stats_by_default(self):
        """Test that read_json() sorts stats automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unsorted_stats = {
                ID_USER: ["user_z", "user_a"],
                ID_SCREEN: ["screen_b", "screen_a"],
                ID_CAMERA: ["cam"],
                ID_MONITOR: ["mon"],
                ID_PLACE: ["place"],
            }

            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(unsorted_stats, f)

            # read_json with default sort_stats=True
            stats = read_json(json_path)

            assert stats[ID_USER] == [
                "user_a",
                "user_z",
            ], f"Expected sorted user list, got {stats[ID_USER]}"
            assert stats[ID_SCREEN] == [
                "screen_a",
                "screen_b",
            ], f"Expected sorted screen list, got {stats[ID_SCREEN]}"

    def test_read_json_sort_stats_false(self):
        """Test that read_json() with sort_stats=False preserves order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unsorted_stats = {
                ID_USER: ["user_z", "user_a"],
                ID_SCREEN: ["screen_b", "screen_a"],
                ID_CAMERA: ["cam"],
                ID_MONITOR: ["mon"],
                ID_PLACE: ["place"],
            }

            json_path = os.path.join(tmpdir, "stats.json")
            with open(json_path, "w") as f:
                json.dump(unsorted_stats, f)

            # read_json with sort_stats=False
            stats = read_json(json_path, sort_stats=False)

            assert stats[ID_USER] == [
                "user_z",
                "user_a",
            ], f"Expected original user order, got {stats[ID_USER]}"
            assert stats[ID_SCREEN] == [
                "screen_b",
                "screen_a",
            ], f"Expected original screen order, got {stats[ID_SCREEN]}"

    def test_ensure_sorted_stats_with_model_wrapper_inputs(self):
        """Test that ensure_sorted_stats produces correct format for ModelWrapper."""
        stats = {
            ID_USER: ["user_z", "user_a"],  # Unsorted!
            ID_SCREEN: ["screen_b", "screen_a"],  # Unsorted!
            ID_CAMERA: ["cam"],
            ID_MONITOR: ["mon"],
            ID_PLACE: ["place"],
        }

        # Ensure stats are sorted before being passed to ModelWrapper
        sorted_stats = ensure_sorted_stats(stats)

        # Verify all ID lists are sorted in the output
        assert sorted_stats[ID_USER] == [
            "user_a",
            "user_z",
        ], f"User IDs not sorted: {sorted_stats[ID_USER]}"
        assert sorted_stats[ID_SCREEN] == [
            "screen_a",
            "screen_b",
        ], f"Screen IDs not sorted: {sorted_stats[ID_SCREEN]}"

        # Verify all required keys are present for ModelWrapper initialization
        required_keys = [ID_USER, ID_SCREEN, ID_CAMERA, ID_MONITOR, ID_PLACE]
        for key in required_keys:
            assert key in sorted_stats, f"Required key {key} missing from sorted stats"
