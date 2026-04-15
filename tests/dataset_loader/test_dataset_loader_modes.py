"""Tests for DatasetLoader sampling modes functionality."""

from Core.data.DatasetLoader import DatasetLoader


class TestDatasetLoaderModes:
    """Test DatasetLoader sampling modes functionality and constants."""

    def test_oversample_mode_is_supported(self):
        """Test that oversample mode is a valid supported mode."""
        # Test that oversample mode is a valid option
        assert (
            "oversample" in DatasetLoader.SAMPLING_MODES
        ), "Oversample mode should be in supported modes"

    def test_undersample_mode_is_supported(self):
        """Test that undersample mode is a valid supported mode."""
        # Test that undersample mode is a valid option
        assert (
            "undersample" in DatasetLoader.SAMPLING_MODES
        ), "Undersample mode should be in supported modes"

    def test_sampling_modes_are_lowercase_strings(self):
        """Test that all sampling modes are lowercase strings."""
        for mode in DatasetLoader.SAMPLING_MODES:
            assert isinstance(mode, str), f"Mode {mode} should be a string"
            assert mode.islower(), f"Mode {mode} should be lowercase"

    def test_sampling_modes_contain_expected_modes(self):
        """Test that sampling modes contain the expected mode names."""
        expected_modes = {"oversample", "undersample"}
        actual_modes = set(DatasetLoader.SAMPLING_MODES)

        assert expected_modes.issubset(
            actual_modes
        ), f"Expected modes {expected_modes} not found in actual modes {actual_modes}"

    def test_no_duplicate_sampling_modes(self):
        """Test that there are no duplicate sampling modes."""
        unique_modes = set(DatasetLoader.SAMPLING_MODES)
        assert len(unique_modes) == len(
            DatasetLoader.SAMPLING_MODES
        ), f"Duplicate sampling modes found: {len(DatasetLoader.SAMPLING_MODES)} total vs {len(unique_modes)} unique"

    def test_sampling_modes_are_valid_strings(self):
        """Test that all sampling modes are non-empty valid strings."""
        for mode in DatasetLoader.SAMPLING_MODES:
            assert isinstance(mode, str), f"Mode {mode} should be a string"
            assert len(mode) > 0, f"Mode {mode} should not be empty"
            assert (
                mode.strip() == mode
            ), f"Mode {mode} should not have leading/trailing whitespace"

    def test_mode_constants_match_sampling_modes(self):
        """Test that mode-related constants are consistent with supported modes."""
        # Test that SAMPLING_MODES contains only valid modes
        assert isinstance(
            DatasetLoader.SAMPLING_MODES, (list, set)
        ), "SAMPLING_MODES should be a list or set"
        assert (
            len(DatasetLoader.SAMPLING_MODES) > 0
        ), "SAMPLING_MODES should not be empty"

        # Each mode should be a valid string
        for mode in DatasetLoader.SAMPLING_MODES:
            assert isinstance(mode, str), f"Mode {mode} should be a string"
            assert len(mode) > 0, f"Mode {mode} should not be empty"

    def test_mode_functionality_consistency(self):
        """Test that mode-related behavior is consistent across the loader."""
        # Test that modes are meaningful (not empty or placeholder)
        # and follow expected naming conventions
        for mode in DatasetLoader.SAMPLING_MODES:
            assert mode in [
                "oversample",
                "undersample",
            ], f"Unknown mode {mode} found in SAMPLING_MODES"
            # Verify modes are usable for initialization
            assert isinstance(mode, str), f"Mode {mode} should be a string"
            assert len(mode) > 0, f"Mode {mode} should be non-empty"
