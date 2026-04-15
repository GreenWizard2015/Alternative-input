"""Tests for SampleFilter trajectory_start() method."""

from Core.data.SampleFilter import SampleFilter
from tests.fixtures.sample_filter_fixtures import _MockStorage


class TestSampleFilterTrajectoryStart:
    """Test trajectory_start() method."""

    def test_trajectory_start_at_start(self, filter_with_1s_window):
        """Test trajectory start for frame at start."""
        # Frame 0 (time=0.0): minInd=0 (no earlier frames within time window)
        min_idx = filter_with_1s_window.trajectory_start(0)
        assert (
            min_idx == 0
        ), "Frame 0 (time=0.0s) should start trajectory at index 0, no earlier frames available"

    def test_trajectory_start_at_middle(self, filter_with_1s_window):
        """Test trajectory start for frame in the middle."""
        # Frame 5 (time=0.5): with maxT=1.0, looks back to time 0.5-1.0=-0.5, starts from frame 0
        min_idx = filter_with_1s_window.trajectory_start(5)
        assert (
            min_idx == 0
        ), "Frame 5 (time=0.5s) should start trajectory at index 0 with maxT=1.0 window"

    def test_trajectory_start_at_end(self, filter_with_1s_window):
        """Test trajectory start for frame at end."""
        # Frame 9 (time=0.9): with maxT=1.0, looks back to time 0.9-1.0=-0.1, starts from frame 0
        min_idx = filter_with_1s_window.trajectory_start(9)
        assert (
            min_idx == 0
        ), "Frame 9 (time=0.9s) should start trajectory at index 0 with maxT=1.0 window"

    def test_trajectory_start_single_frame(self, storage_single_frame):
        """Test trajectory start with single frame."""
        filter_obj = SampleFilter(storage_single_frame, minFrames=1, maxT=1.0)
        min_idx = filter_obj.trajectory_start(0)
        assert min_idx == 0, "Single frame at index 0 should start trajectory at itself"

    def test_trajectory_start_smaller_window(self, filter_with_1s_window):
        """Test trajectory start with smaller time window."""
        storage = _MockStorage([0.0, 0.5, 1.0, 1.5, 2.0])
        filter_obj = SampleFilter(storage, minFrames=1, maxT=0.5)
        # Frame 2 (time=1.0): window [0.5, 1.5] includes frames 1,2 (times 0.5, 1.0)
        min_idx = filter_obj.trajectory_start(2)
        assert (
            min_idx == 1
        ), f"Frame 2 (time=1.0s) with maxT=0.5 should start at index 1 (time=0.5s), got {min_idx}"

    def test_trajectory_start_larger_window(self, storage_10_frames):
        """Test trajectory start with larger time window."""
        filter_obj = SampleFilter(storage_10_frames, minFrames=1, maxT=2.0)
        # Frame 5 (time=0.5): ±2s window [-1.5, 2.5] includes all frames 0-9, starts from 0
        min_idx = filter_obj.trajectory_start(5)
        assert (
            min_idx == 0
        ), f"Frame 5 (time=0.5s) with maxT=2.0 should start at index 0, got {min_idx}"
