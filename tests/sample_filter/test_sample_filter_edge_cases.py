"""Tests for SampleFilter edge cases and boundary conditions."""

from Core.data.SampleFilter import SampleFilter


class TestSampleFilterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_frame_storage(self, storage_single_frame):
        """Test SampleFilter with single frame."""
        filter_obj = SampleFilter(storage_single_frame, minFrames=1, maxT=0.5)
        assert filter_obj.isValid(
            0
        ), "Single frame at time 0.0s should be valid with minFrames=1, maxT=0.5"

    def test_large_time_window(self, storage_sparse_frames):
        """Test SampleFilter with time window larger than all data."""
        filter_obj = SampleFilter(storage_sparse_frames, minFrames=2, maxT=10.0)
        # All frames should start from frame 0 in trajectory for any frame with large time window
        for idx in range(len(storage_sparse_frames.times)):
            min_idx = filter_obj.trajectory_start(idx)
            assert (
                min_idx == 0
            ), f"Frame {idx} (time={storage_sparse_frames.times[idx]}s) should start trajectory at index 0 with maxT=10.0 window, got {min_idx}"

    def test_tiny_time_window(self, storage_10_frames):
        """Test SampleFilter with very small time window."""
        filter_obj = SampleFilter(storage_10_frames, minFrames=1, maxT=0.05)
        # With 0.1s intervals and 0.05s window, only frame itself should start the range
        min_idx = filter_obj.trajectory_start(5)
        assert (
            min_idx == 5
        ), f"Frame 5 (time=0.5s) with maxT=0.05 should start at itself (index 5), window too small for other frames, got {min_idx}"

    def test_high_min_frames_requirement(self, storage_10_frames):
        """Test SampleFilter with high minFrames requirement."""
        filter_obj = SampleFilter(storage_10_frames, minFrames=10, maxT=1.0)
        # Only last frame has all 10 frames in its trajectory within 1s window
        assert not filter_obj.isValid(
            0
        ), "Frame 0 should be invalid with minFrames=10, only 1 frame available in 1s window"
        assert filter_obj.isValid(
            9
        ), "Frame 9 should be valid with minFrames=10, has all 10 frames available in 1s window"
