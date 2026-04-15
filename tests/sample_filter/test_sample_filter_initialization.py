"""Tests for SampleFilter initialization and validation."""

from Core.data.SampleFilter import SampleFilter


class TestSampleFilterInitialization:
    """Test SampleFilter initialization and validation."""

    def test_valid_initialization_affects_validity_checks(self, storage_10_frames):
        """Test that SampleFilter initialization with valid parameters affects validation.

        Verifies the initialization is working by checking that validity
        checks are properly constrained by the minFrames parameter.
        """
        filter_obj = SampleFilter(storage_10_frames, minFrames=3, maxT=1.0)
        # Frame 2 has exactly 3 frames (0-2), should be valid at minFrames=3
        assert filter_obj.isValid(
            2
        ), f"Frame 2 should be valid with minFrames=3, storage times: {[i * 0.1 for i in range(3)]}"

    def test_initialization_min_frames_affects_validity(self, storage_10_frames):
        """Test that different minFrames values affect which samples are valid.

        Verifies that increasing minFrames requirement makes earlier frames invalid,
        demonstrating that initialization parameters have real behavioral impact.
        """
        # With minFrames=1, frame 0 should be valid (has itself)
        filter_1 = SampleFilter(storage_10_frames, minFrames=1, maxT=1.0)
        assert filter_1.isValid(
            0
        ), f"Frame 0 should be valid with minFrames=1, frame time: {0.0}s"

        # With minFrames=3, frame 0 should be invalid (not enough history - only 1 frame)
        filter_3 = SampleFilter(storage_10_frames, minFrames=3, maxT=1.0)
        assert not filter_3.isValid(
            0
        ), "Frame 0 should be invalid with minFrames=3, only has 1 frame within 1s window"

    def test_initialization_max_t_affects_trajectory_range(self, storage_10_frames):
        """Test that different maxT values affect trajectory start calculations.

        Verifies that increasing maxT window size affects where the trajectory starts,
        allowing earlier frames to be included in the trajectory window.
        """
        # With maxT=0.1, only nearby frames are included
        filter_small = SampleFilter(storage_10_frames, minFrames=1, maxT=0.1)
        min_small = filter_small.trajectory_start(5)

        # With maxT=2.0, many more frames are included
        filter_large = SampleFilter(storage_10_frames, minFrames=1, maxT=2.0)
        min_large = filter_large.trajectory_start(5)

        # Larger time window should start further back (at or before smaller window)
        assert (
            min_large <= min_small
        ), f"Larger maxT=2.0 should start earlier or same as maxT=0.1: got min_large={min_large}, min_small={min_small}, time difference should be >= 0"
