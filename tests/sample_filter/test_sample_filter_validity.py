"""Tests for SampleFilter isValid() method."""

from Core.data.SampleFilter import SampleFilter


class TestSampleFilterValidity:
    """Test isValid() method for sample validation."""

    def test_valid_sample_with_enough_frames(self, filter_with_1s_window):
        """Test that sample with enough frames in time window is valid."""
        # Frame 9 (time=0.9) has frames 0-9 within 1 second window
        # That's 10 frames, exceeds minFrames=3 requirement
        assert filter_with_1s_window.isValid(
            9
        ), "Frame 9 (time=0.9s) should be valid with 10 frames available in 1s window, minFrames=3"

    def test_valid_sample_at_middle(self, filter_with_1s_window):
        """Test validity of frame in the middle."""
        # Frame 5 (time=0.5) has frames from ~0.0 to 0.9 within 1s window (10 frames total)
        assert filter_with_1s_window.isValid(
            5
        ), "Frame 5 (time=0.5s) should be valid with multiple frames available in 1s window, minFrames=3"

    def test_invalid_sample_at_start(self, filter_with_1s_window):
        """Test that early frames without enough history are invalid."""
        # Frame 0 (time=0.0) has only 1 frame (itself), less than minFrames=3 requirement
        assert not filter_with_1s_window.isValid(
            0
        ), "Frame 0 (time=0.0s) should be invalid with only 1 frame available in 1s window, minFrames=3"

    def test_invalid_sample_with_insufficient_frames(self, filter_with_1s_window):
        """Test that sample with fewer than minFrames is invalid."""
        # Frame 1 (time=0.1) has frames 0-1, only 2 frames, less than minFrames=3 requirement
        assert not filter_with_1s_window.isValid(
            1
        ), "Frame 1 (time=0.1s) should be invalid with only 2 frames available in 1s window, minFrames=3"

    def test_boundary_valid_at_min_frames(self, storage_10_frames):
        """Test sample becomes valid at exactly minFrames threshold."""
        filter_obj = SampleFilter(storage_10_frames, minFrames=3, maxT=1.0)
        # Frame 2 (time=0.2) has frames 0-2, exactly 3 frames = minFrames boundary
        assert filter_obj.isValid(
            2
        ), "Frame 2 (time=0.2s) should be valid at minFrames threshold with exactly 3 frames (0-2) in 1s window"

    def test_sparse_storage_validity(self, storage_sparse_frames):
        """Test validity checking with sparse frames."""
        filter_obj = SampleFilter(storage_sparse_frames, minFrames=2, maxT=1.0)
        # Frame 2 (time=1.0) has frames 0-2 within 1s window: times 0.0, 0.5, 1.0 (3 frames >= minFrames=2)
        assert filter_obj.isValid(
            2
        ), "Frame 2 (time=1.0s) should be valid with sparse storage having 3 frames within 1s window, minFrames=2"
