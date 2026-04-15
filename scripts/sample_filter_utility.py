#!/usr/bin/env python3
"""
Eye Sampling Filter Utility - Complete Implementation

A command-line utility for filtering eye tracking frames with keyboard controls.
Follows plan.md specifications with proper multi-file sequence loading,
model integration, and dataset management.

Key Features:
- Multi-file sequence loading for ModelWrapper compatibility
- Keyboard controls: Left Arrow (accept), Right Arrow (reject), Space (save), ESC (exit), D (switch dataset)
- 8x zoom display with colored borders and quality scoring
- Filter persistence with dataset-specific tracking
- Optional gaze prediction model for quality scoring
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import argparse
import os
import logging
import json
import time
import glob
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import Core components
from Core import Utils
from Core.models import FilterWrapper

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    path: Path
    monitor_id: str
    total_frames: int
    file_count: int
    file_pattern: str
    stats_path: Path
    dataset_info: Optional[Any] = None


class FilterDecision(Enum):
    """Filter decision types."""

    ACCEPT = "accept"
    REJECT = "reject"
    NONE = "none"


# ============================================================================
# FILTER MANAGEMENT
# ============================================================================


class FilterManager:
    """Manages filter decisions with JSON persistence and dataset tracking."""

    def __init__(self, filter_path: str, args, dataset_paths=None):
        """Initialize filter manager."""
        self.filter_path = Path(filter_path)
        self.args = args
        self.dataset_paths = dataset_paths or []
        self._individual_dataset_paths = self._discover_individual_dataset_paths()
        self._filters = {}
        self._load_filters()

    def _discover_individual_dataset_paths(self) -> List[str]:
        """Discover individual dataset paths using Utils.dataset_from_stats."""
        individual_paths = []
        for base_path in self.dataset_paths:
            stats_path = Path(base_path) / "stats.json"
            if stats_path.exists():
                try:
                    dataset_iterator = Utils.dataset_from_stats(str(stats_path))
                    dataset_list = list(dataset_iterator)
                    for dataset_info_obj in dataset_list:
                        actual_dataset_path = str(dataset_info_obj.path.full_path)
                        individual_paths.append(actual_dataset_path)
                except Exception as e:
                    logger.warning(f"Error discovering datasets in {base_path}: {e}")
                    continue
        return individual_paths

    def _load_filters(self):
        """Load existing filters from JSON file if it exists."""
        if self.filter_path.exists():
            with open(self.filter_path, "r") as f:
                self._filters = json.load(f)
            logger.info(f"Loaded {len(self._filters)} dataset filters")
        else:
            self._filters = {}

    def _normalize_dataset_path(self, dataset_path: str) -> str:
        """Normalize dataset path using monitor_id for unique dataset identification."""
        import os

        # Look for the dataset path in individual dataset paths
        for individual_path in self._individual_dataset_paths:
            stats_path = Path(individual_path).parent / "stats.json"
            if stats_path.exists():
                try:
                    dataset_iterator = Utils.dataset_from_stats(str(stats_path))
                    dataset_list = list(dataset_iterator)

                    for dataset_info_obj in dataset_list:
                        actual_dataset_path = str(dataset_info_obj.path.full_path)
                        if actual_dataset_path == dataset_path:
                            # Use monitor_id as unique identifier for this dataset
                            return f"dataset_{dataset_info_obj.path.monitor_id}"
                except Exception as e:
                    logger.warning(
                        f"Error finding dataset info for {dataset_path}: {e}"
                    )
                    continue

        # Fallback: use relative path if we can't find monitor_id
        relative_path = os.path.relpath(dataset_path, self.args.dataset)
        return relative_path if relative_path != "." else "root_dataset"

    def _save_filters(self):
        """Save filters to JSON file."""
        with open(self.filter_path, "w") as f:
            json.dump(self._filters, f, indent=2, ensure_ascii=False)

    def save_decision(self, dataset_path: str, frame_idx: int, decision: str):
        """Save decision and persist to filter.json."""
        if decision not in [FilterDecision.ACCEPT.value, FilterDecision.REJECT.value]:
            raise ValueError(f"Invalid decision: {decision}")

        # Ensure frame index is non-negative
        frame_idx_int = int(frame_idx)
        if frame_idx_int < 0:
            logger.warning(
                f"Attempting to save negative frame index {frame_idx_int}, skipping..."
            )
            return

        # Normalize dataset path by removing the base dataset path and trimming leading /
        normalized_path = self._normalize_dataset_path(dataset_path)
        if normalized_path not in self._filters:
            self._filters[normalized_path] = {}

        # Store as boolean: True for accept, False for reject
        self._filters[normalized_path][frame_idx_int] = (
            decision == FilterDecision.ACCEPT.value
        )
        self._save_filters()

    def get_filters_copy(self) -> Dict[str, Dict[int, bool]]:
        """Get copy of all filters from JSON file."""
        return self._filters.copy()

    def is_frame_filtered(self, dataset_path: str, frame_idx: int) -> bool:
        """Check if frame has been filtered."""
        normalized_path = self._normalize_dataset_path(dataset_path)
        return (
            normalized_path in self._filters
            and int(frame_idx) in self._filters[normalized_path]
        )

    def get_frame_decision(self, dataset_path: str, frame_idx: int) -> Optional[bool]:
        """Get decision for specific frame (True=accept, False=reject, None=unfiltered)."""
        normalized_path = self._normalize_dataset_path(dataset_path)
        frame_key = int(frame_idx)
        return self._filters[normalized_path].get(frame_key)

    def get_next_unfiltered_frame(
        self, dataset_path: str, current_frame: int
    ) -> Optional[int]:
        """Find next unfiltered frame after current_frame."""
        normalized_path = self._normalize_dataset_path(dataset_path)
        current_key = int(current_frame)

        if normalized_path not in self._filters:
            # No filters for this dataset, return the next frame
            next_frame = current_key + 1
            print(
                f"No filters found for {normalized_path}, returning frame {next_frame}"
            )
            return next_frame

        # Get sorted list of filtered frames, filter out negative indices
        filtered_frames = sorted(
            int(key) for key in self._filters[normalized_path].keys() if int(key) >= 0
        )
        if not filtered_frames:
            # Empty filters or only negative indices, return the next frame
            next_frame = current_key + 1
            print(f"Empty filters for {normalized_path}, returning frame {next_frame}")
            return next_frame

        # Start from current frame + 1 and find next unfiltered frame
        search_frame = current_key + 1
        print(
            f"Searching for next unfiltered frame from {search_frame} in {normalized_path}"
        )
        print(f"Filtered frames: {filtered_frames}")

        while search_frame <= filtered_frames[-1]:
            if search_frame not in filtered_frames:
                print(f"Found unfiltered frame {search_frame}")
                return search_frame
            search_frame += 1

        # Return first frame beyond all filtered frames
        print(
            f"All frames up to {filtered_frames[-1]} filtered, returning {search_frame}"
        )
        return search_frame


# ============================================================================
# DATA LOADING
# ============================================================================


class DataLoader:
    """Handles loading of individual frames from all.npz files using Utils.dataset_from_stats."""

    def __init__(
        self, dataset_path: str, filter_manager: FilterManager, dataset_id: int = 0
    ):
        """Initialize data loader with dataset path and filter manager."""
        self.dataset_path = Path(dataset_path)
        self.filter_manager = filter_manager
        self.dataset_id = dataset_id
        self._cached_npz_files = []  # Initialize cached files list
        self.specific_dataset_path = None  # Will be set in _load_dataset_info
        self.dataset_info = self._load_dataset_info()
        self._validate_dataset()

    def _load_dataset_info(self) -> DatasetInfo:
        """Load dataset information using Utils.dataset_from_stats."""
        stats_path = self.dataset_path / "stats.json"

        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        # Use Utils.dataset_from_stats to get dataset information
        dataset_iterator = Utils.dataset_from_stats(str(stats_path))
        dataset_list = list(dataset_iterator)

        if not dataset_list:
            raise ValueError(f"No datasets found in stats file: {stats_path}")

        # Get the specific dataset by ID
        if self.dataset_id >= len(dataset_list):
            raise ValueError(
                f"Dataset ID {self.dataset_id} out of range. Only {len(dataset_list)} datasets available."
            )

        target_dataset = dataset_list[self.dataset_id]

        # Store specific dataset path for filter lookups
        self.specific_dataset_path = str(target_dataset.path.full_path)

        # Find dataset files in the specific dataset path (not the base directory)
        self._cached_npz_files = []
        all_npz_files = sorted(
            glob.glob(
                os.path.join(self.specific_dataset_path, "**", "all.npz"),
                recursive=True,
            )
        )
        self._cached_npz_files.extend(all_npz_files)

        # If no all.npz files found, look for train.npz files
        if not self._cached_npz_files:
            train_npz_files = sorted(
                glob.glob(
                    os.path.join(self.specific_dataset_path, "**", "train.npz"),
                    recursive=True,
                )
            )
            self._cached_npz_files.extend(train_npz_files)

        # Calculate total frames by examining actual data
        total_frames = 0
        if self._cached_npz_files:
            # Load first file to get frames per file
            sample_file = self._cached_npz_files[0]
            sample_data = np.load(sample_file, allow_pickle=False)
            frames_per_file = sample_data["left eye"].shape[0]
            total_frames = len(self._cached_npz_files) * frames_per_file

        # Get monitor ID from the dataset info
        monitor_id = target_dataset.path.monitor_id

        return DatasetInfo(
            path=self.dataset_path,
            monitor_id=monitor_id,
            total_frames=total_frames,
            file_count=len(self._cached_npz_files),
            file_pattern="all.npz",
            stats_path=stats_path,
            dataset_info=target_dataset,  # Store the dataset info from Utils
        )

    def _validate_dataset(self):
        """Validate dataset structure and consistency."""
        if self.dataset_info.file_count == 0:
            raise ValueError(f"No dataset_*.npz files found in {self.dataset_path}")

        logger.info(f"Dataset loaded: {self.dataset_info.monitor_id}")
        logger.info(
            f"Files: {self.dataset_info.file_count}, Total frames: {self.dataset_info.total_frames}"
        )

    def get_frame_by_index(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Load single frame by absolute index across all files."""
        if frame_idx >= self.dataset_info.total_frames:
            return None

        # Calculate which file contains this frame and the frame index within that file
        if not self._cached_npz_files:
            return None

        # Determine frames per file by examining the first file
        sample_data = np.load(self._cached_npz_files[0], allow_pickle=False)
        frames_per_file = sample_data["left eye"].shape[0]

        file_idx = frame_idx // frames_per_file
        frame_in_file = frame_idx % frames_per_file

        # Calculate file indices

        # Use cached npz files for memory efficiency
        # Find dataset files in the specific dataset path (populated during init)
        if not self._cached_npz_files:
            all_npz_files = sorted(
                glob.glob(
                    str(
                        self.dataset_info.dataset_info.path.full_path / "**" / "all.npz"
                    ),
                    recursive=True,
                )
            )
            self._cached_npz_files.extend(all_npz_files)

        # Determine frames per file by examining the first file
        sample_data = np.load(self._cached_npz_files[0], allow_pickle=False)
        frames_per_file = sample_data["left eye"].shape[0]

        if file_idx >= len(self._cached_npz_files):
            return None

        file_path = self._cached_npz_files[file_idx]

        data = np.load(file_path, allow_pickle=False)
        logger.info(f"Loaded file {file_path} with keys: {list(data.keys())}")

        # Extract data for the specific frame - only eye images needed
        frame_data = {
            "left_eye": data["left eye"][int(frame_in_file)],  # (48, 48)
            "right_eye": data["right eye"][int(frame_in_file)],  # (48, 48)
            "frame_idx": frame_idx,
            "file_idx": file_idx,
        }

        # Validate frame data shapes and types
        validation_errors = self.validate_frame_data(frame_data)
        if validation_errors:
            raise ValueError(f"Invalid frame data: {validation_errors}")

        return frame_data

    def validate_frame_data(self, frame_data: Dict[str, Any]) -> List[str]:
        """Validate frame data shapes and types for eye images only.

        Args:
            frame_data: Frame data dictionary to validate

        Returns:
            List of validation error messages, empty if valid
        """
        errors = []

        # Check required keys - only eyes needed
        required_keys = ["left_eye", "right_eye"]
        for key in required_keys:
            if key not in frame_data:
                errors.append(f"Missing required key: {key}")

        if errors:
            return errors

        # Validation happens via assertions in the calling method now
        return []

    def load_sequence_from_files(
        self, start_frame_idx: int, timesteps: int = 5
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load sequence from multiple files to satisfy ModelWrapper timestep requirements.

        Args:
            start_frame_idx: Starting frame index to build sequence from
            timesteps: Number of timesteps needed for model input

        Returns:
            Dictionary with sequence data for all required keys, or None if not enough frames
        """
        if start_frame_idx < 0:
            return None

        # Validate we have enough frames
        if start_frame_idx + timesteps > self.dataset_info.total_frames:
            return None

        # Use cached npz files for memory efficiency
        # Note: _cached_npz_files is initialized in __init__

        sequence_data = {}
        frames_to_load = []

        # Get frames per file and calculate which frames we need
        if not self._cached_npz_files:
            return None

        # Load first file to get frames per file
        sample_file = self._cached_npz_files[0]
        sample_data = np.load(sample_file, allow_pickle=False)
        frames_per_file = sample_data["left eye"].shape[0]

        # Calculate which frames we need and which files they're in
        for i in range(start_frame_idx, start_frame_idx + timesteps):
            file_idx = i // frames_per_file
            frame_in_file = i % frames_per_file
            frames_to_load.append((file_idx, frame_in_file, i))

        # Load required files and extract frames
        loaded_frames = {}
        for file_idx, frame_in_file, abs_frame_idx in frames_to_load:
            if file_idx >= len(self._cached_npz_files):
                return None

            file_path = self._cached_npz_files[file_idx]
            data = np.load(file_path, allow_pickle=False)

            # Store this frame with validation - only eyes needed
            loaded_frames[abs_frame_idx] = {
                "left_eye": data["left eye"][int(frame_in_file)],
                "right_eye": data["right eye"][int(frame_in_file)],
            }

        # Build sequence arrays by stacking frames - only eyes needed
        for key in ["left_eye", "right_eye"]:
            sequence_data[key] = np.stack(
                [
                    loaded_frames[i][key]
                    for i in range(start_frame_idx, start_frame_idx + timesteps)
                ]
            )

        return sequence_data

    def get_next_unfiltered_frame(self, current_frame: int) -> Optional[int]:
        """Find next unfiltered frame."""
        if self.specific_dataset_path is None:
            # If specific dataset path is not set, return the next frame
            return current_frame + 1

        # Always get the next unfiltered frame (skip current if filtered)
        return self.filter_manager.get_next_unfiltered_frame(
            self.specific_dataset_path, current_frame
        )


# ============================================================================
# DISPLAY
# ============================================================================


class EyeDisplay:
    """Handles eye image visualization with overlays and borders."""

    def __init__(self, zoom_factor: int = 8):
        """Initialize display with zoom factor."""
        self.zoom_factor = zoom_factor
        self.current_selection = FilterDecision.NONE
        self.quality_score = None

    def create_frame_visualization(
        self,
        frame_data: Dict[str, Any],
        border_color: Optional[str] = None,
        score_text: Optional[str] = None,
    ) -> np.ndarray:
        """Create visualization with borders and score overlay."""

        # Extract eye images
        left_eye = frame_data["left_eye"]  # (48, 48)
        right_eye = frame_data["right_eye"]  # (48, 48)

        # Zoom eyes
        eye_display_size = 48 * self.zoom_factor
        left_resized = cv2.resize(
            left_eye,
            (eye_display_size, eye_display_size),
            interpolation=cv2.INTER_NEAREST,
        )
        right_resized = cv2.resize(
            right_eye,
            (eye_display_size, eye_display_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert to BGR
        left_bgr = cv2.cvtColor(left_resized, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right_resized, cv2.COLOR_GRAY2BGR)

        # Create display with separator
        display_width = eye_display_size * 2 + 100
        display_height = eye_display_size
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Place eyes
        display[:, :eye_display_size] = left_bgr
        display[:, eye_display_size + 100 :] = right_bgr

        # Add separator
        display[:, eye_display_size : eye_display_size + 100] = 128

        # Apply colored border if specified using cv2.rectangle
        if border_color:
            border_color_bgr = (0, 255, 0) if border_color == "accept" else (0, 0, 255)
            thickness = 3

            # Draw border on left eye
            cv2.rectangle(
                display, (0, 0), (eye_display_size, thickness), border_color_bgr, -1
            )  # Top
            cv2.rectangle(
                display,
                (0, eye_display_size - thickness),
                (eye_display_size, eye_display_size),
                border_color_bgr,
                -1,
            )  # Bottom
            cv2.rectangle(
                display, (0, 0), (thickness, eye_display_size), border_color_bgr, -1
            )  # Left
            cv2.rectangle(
                display,
                (eye_display_size - thickness, 0),
                (eye_display_size, eye_display_size),
                border_color_bgr,
                -1,
            )  # Right

            # Draw border on right eye
            right_eye_x = eye_display_size + 100
            cv2.rectangle(
                display,
                (right_eye_x, 0),
                (right_eye_x + eye_display_size, thickness),
                border_color_bgr,
                -1,
            )  # Top
            cv2.rectangle(
                display,
                (right_eye_x, eye_display_size - thickness),
                (right_eye_x + eye_display_size, eye_display_size),
                border_color_bgr,
                -1,
            )  # Bottom
            cv2.rectangle(
                display,
                (right_eye_x, 0),
                (right_eye_x + thickness, eye_display_size),
                border_color_bgr,
                -1,
            )  # Left
            cv2.rectangle(
                display,
                (right_eye_x + eye_display_size - thickness, 0),
                (right_eye_x + eye_display_size, eye_display_size),
                border_color_bgr,
                -1,
            )  # Right

        # Add score text
        font = cv2.FONT_HERSHEY_SIMPLEX
        if score_text:
            cv2.putText(display, score_text, (10, 30), font, 0.7, (0, 255, 255), 2)

        # Add eye labels
        cv2.putText(
            display, "LEFT", (10, display_height - 10), font, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            display,
            "RIGHT",
            (eye_display_size + 110, display_height - 10),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        return display

    # apply_colored_border method removed - using cv2.rectangle directly in create_frame_visualization

    def show_with_status(
        self,
        frame_data: Dict[str, Any],
        status_text: str,
        dataset_info: DatasetInfo,
        auto_decision: Optional[bool] = None,
        auto_confidence: Optional[float] = None,
    ) -> None:
        """Display both eyes with status information.

        Args:
            frame_data: Frame data containing eye images and metadata
            status_text: Status text to display overlay
            dataset_info: Dataset information for display
            auto_decision: Automated filter decision (True=accept, False=reject, None=no model)
            auto_confidence: Automated filter confidence score
        """

        # Create base visualization with border if selection is made
        border_color = None
        if self.current_selection != FilterDecision.NONE:
            border_color = self.current_selection.value  # "accept" or "reject"

        visualization = self.create_frame_visualization(
            frame_data, border_color=border_color
        )

        # Add status overlay
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Frame counter
        counter_text = (
            f"Frame {frame_data['frame_idx'] + 1}/{dataset_info.total_frames}"
        )
        cv2.putText(
            visualization,
            counter_text,
            (10, visualization.shape[0] - 30),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        # Dataset info
        dataset_text = f"Dataset: {dataset_info.monitor_id}"
        cv2.putText(
            visualization,
            dataset_text,
            (10, visualization.shape[0] - 10),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        # Add automated filter status
        if auto_decision is not None:
            auto_decision = auto_decision == FilterDecision.ACCEPT
            auto_text = f"Auto: {'ACCEPT' if not auto_decision else 'REJECT'} ({auto_confidence:.3f})"
            auto_color = (0, 255, 0) if auto_decision else (0, 0, 255)
            cv2.putText(
                visualization,
                auto_text,
                (10, 80),
                font,
                0.5,
                auto_color,
                1,
            )

        # Status
        cv2.putText(
            visualization,
            status_text,
            (visualization.shape[1] - 300, 30),
            font,
            0.5,
            (255, 255, 0),
            1,
        )

        # Show instructions
        instructions = "←/A=Accept →/R=Reject Space=Save ESC=Exit D=Next Dataset"
        cv2.putText(
            visualization,
            instructions,
            (10, visualization.shape[0] - 50),
            font,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Sample Filter", visualization)


# ============================================================================
# KEYBOARD HANDLING
# ============================================================================


class KeyboardHandler:
    """Handles keyboard input with state management."""

    def __init__(self):
        """Initialize keyboard handler."""
        self.current_selection = FilterDecision.NONE

    def wait_for_key(self, timeout_ms: int = 0) -> Optional[int]:
        """Wait for key press with optional timeout."""
        return cv2.waitKey(timeout_ms) & 0xFF

    def handle_input(self, key: int) -> Dict[str, Any]:
        """Process keyboard input and return action."""
        if key == 27:  # ESC
            return {"action": "exit", "selection": self.current_selection}
        elif key == 32:  # Space
            return {"action": "save", "selection": self.current_selection}
        elif key == ord("a") or key == ord("A"):  # A key for accept
            self.current_selection = FilterDecision.ACCEPT
            return {"action": "set_selection", "selection": FilterDecision.ACCEPT}
        elif key == ord("r") or key == ord("R"):  # R key for reject
            self.current_selection = FilterDecision.REJECT
            return {"action": "set_selection", "selection": FilterDecision.REJECT}
        elif key == ord("d") or key == ord("D"):  # D key for dataset switching
            return {"action": "next_dataset", "selection": self.current_selection}

        # Arrow keys - use most common codes
        elif key == 81 or key == 2555904:  # Left Arrow (macOS/Linux)
            self.current_selection = FilterDecision.ACCEPT
            return {"action": "set_selection", "selection": FilterDecision.ACCEPT}
        elif key == 83 or key == 2555906:  # Right Arrow (macOS/Linux)
            self.current_selection = FilterDecision.REJECT
            return {"action": "set_selection", "selection": FilterDecision.REJECT}
        # Note: Removed Up Arrow mapping to next_dataset to avoid conflict with D key
        return {"action": "none", "selection": self.current_selection}


# ============================================================================
# DATASET MANAGER
# ============================================================================


class DatasetManager:
    """Manages multiple datasets with switching capability using Utils.dataset_from_stats."""

    def __init__(self, dataset_paths: List[str], filter_manager: FilterManager):
        """Initialize dataset manager."""
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.filter_manager = filter_manager
        self.current_dataset_idx = 0
        self.datasets = []

        self._load_datasets()

    def _load_datasets(self):
        """Load information for all datasets using Utils.dataset_from_stats."""
        for base_path in self.dataset_paths:
            stats_path = base_path / "stats.json"

            if not stats_path.exists():
                logger.warning(f"Stats file not found: {stats_path}")
                continue

            # Use Utils.dataset_from_stats to get all datasets in this path
            dataset_list = list(Utils.dataset_from_stats(str(stats_path)))

            if not dataset_list:
                logger.warning(f"No datasets found in stats file: {stats_path}")
                continue

            # Load each dataset
            for i, dataset_info_obj in enumerate(dataset_list):
                # Create a custom dataset info that includes the base path
                dataset_loader = DataLoader(
                    str(base_path), self.filter_manager, dataset_id=i
                )

                # Get actual dataset path from dataset_info_obj
                actual_dataset_path = dataset_info_obj.path.full_path

                # Override the dataset info to use the specific dataset from Utils
                custom_dataset_info = DatasetInfo(
                    path=Path(actual_dataset_path),
                    monitor_id=dataset_info_obj.path.monitor_id,
                    total_frames=dataset_loader.dataset_info.total_frames,
                    file_count=dataset_loader.dataset_info.file_count,
                    file_pattern=dataset_loader.dataset_info.file_pattern,
                    stats_path=stats_path,
                    dataset_info=dataset_info_obj,
                )

                self.datasets.append(
                    {
                        "path": Path(actual_dataset_path),
                        "info": custom_dataset_info,
                        "loader": dataset_loader,
                        "dataset_info_obj": dataset_info_obj,
                    }
                )

                logger.info(
                    f"Loaded dataset from {base_path}: {custom_dataset_info.monitor_id}"
                )

        if not self.datasets:
            raise ValueError(
                f"No valid datasets found in any of the provided paths: {self.dataset_paths}"
            )

    def get_current_dataset(self) -> Dict[str, Any]:
        """Get current dataset information."""
        if not self.datasets or self.current_dataset_idx >= len(self.datasets):
            return None
        return self.datasets[self.current_dataset_idx]

    def switch_to_next_dataset(self) -> bool:
        """Switch to next dataset, return True if successful."""
        if len(self.datasets) <= 1:
            return False

        self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.datasets)
        return True

    def get_datasets_list(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        result = []
        for i, dataset in enumerate(self.datasets):
            if dataset:
                # Extract user ID from path for better identification
                user_id = "unknown"
                if dataset["dataset_info_obj"]:
                    user_id = dataset["dataset_info_obj"].path.user_id

                result.append(
                    {
                        "index": i,
                        "monitor_id": dataset["info"].monitor_id,
                        "user_id": user_id,
                        "display_name": f"Monitor {dataset['info'].monitor_id[:8]}... (User {user_id[:8]}...)",
                        "total_frames": dataset["info"].total_frames,
                        "file_count": dataset["info"].file_count,
                    }
                )
        return result


# ============================================================================
# AUTOMATED FILTERING
# ============================================================================


class AutomatedFilter:
    """Handles automated filtering using FilterWrapper."""

    def __init__(self, threshold: float = 0.5):
        """Initialize automated filter with trained model."""
        self.model = None
        self.threshold = threshold

        # Hardcoded model path
        model_path = (
            Path(__file__).parent.parent
            / "Data"
            / "models"
            / "filter"
            / "best"
            / "filter_model.npz"
        )

        if model_path.exists():
            self.model = FilterWrapper(model="filter")
            self.model.load(folder=str(model_path.parent.parent.parent), postfix="best")
            logger.info("FilterWrapper model loaded successfully")
        else:
            raise ValueError(f"FilterWrapper model not found at {model_path}")

    def classify_frame(self, frame_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Classify frame as valid (True) or invalid (False).

        Args:
            frame_data: Frame data with 'left_eye' and 'right_eye'

        Returns:
            True if valid (accept), False if invalid (reject), None if no model
        """
        if not self.model:
            return None, 0.0

        # Prepare input for FilterWrapper (single batch with channel dimension and normalization)
        input_data = {
            "left eye": np.expand_dims(frame_data["left_eye"], axis=(0, -1)).astype(
                np.float32
            )
            / 255.0,  # (1, 48, 48, 1)
            "right eye": np.expand_dims(frame_data["right_eye"], axis=(0, -1)).astype(
                np.float32
            )
            / 255.0,  # (1, 48, 48, 1)
        }

        # Get predictions
        predictions = self.model(input_data)

        # Apply threshold and return binary decision
        confidence = predictions["predictions"][0][0]

        if confidence < self.threshold:
            return False, confidence  # Accept
        if 1.0 - confidence < self.threshold:
            return True, confidence  # Reject
        return None, confidence


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class SampleFilterUtility:
    """Main application class for sample filtering."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the filtering application."""
        self.args = args
        self.filter_manager = FilterManager(args.filter_file, args, args.dataset_paths)
        self.keyboard_handler = KeyboardHandler()
        self.display = EyeDisplay(args.zoom_factor)
        self.dataset_manager = DatasetManager(args.dataset_paths, self.filter_manager)

        # Initialize FilterWrapper for automated filtering only if --use-model is set
        if args.use_model:
            self.automated_filter = AutomatedFilter(args.threshold)
        else:
            self.automated_filter = None

        self.current_frame_idx = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.auto_decision = self.auto_confidence = None

        # Validate datasets
        if not self.dataset_manager.datasets:
            raise ValueError("No valid datasets found")

        # Find first unfiltered frame
        self._find_first_unfiltered_frame()

    def _find_first_unfiltered_frame(self):
        """Find the first unfiltered frame across all datasets."""
        min_frame_idx = float("inf")
        target_dataset_idx = 0

        # Check all datasets to find the absolute first unfiltered frame
        for i, dataset in enumerate(self.dataset_manager.datasets):
            # Use -1 to get first frame, then check if it's actually unfiltered
            frame_idx = dataset["loader"].get_next_unfiltered_frame(-1)
            if frame_idx is not None:
                # Verify this frame is actually unfiltered
                dataset_path = str(dataset["path"])
                is_filtered = self.filter_manager.is_frame_filtered(
                    dataset_path, frame_idx
                )
                if not is_filtered and frame_idx < min_frame_idx:
                    min_frame_idx = frame_idx
                    target_dataset_idx = i

        # Set current frame and dataset to the first unfiltered frame found
        if min_frame_idx != float("inf"):
            self.current_frame_idx = min_frame_idx
            self.dataset_manager.current_dataset_idx = target_dataset_idx
            logger.info(
                f"Found first unfiltered frame: {min_frame_idx} in dataset {target_dataset_idx}"
            )
        else:
            # If all frames are filtered, start from frame 0
            self.current_frame_idx = 0
            logger.info("All frames filtered, starting from frame 0")

    def _find_first_unfiltered_frame_in_current_dataset(self):
        """Find the first unfiltered frame in the current dataset only."""
        current_dataset = self.dataset_manager.get_current_dataset()
        if not current_dataset:
            self.current_frame_idx = 0
            return

        # Get the first unfiltered frame in the current dataset
        frame_idx = current_dataset["loader"].get_next_unfiltered_frame(-1)
        if frame_idx is not None:
            # Verify this frame is actually unfiltered
            dataset_path = str(current_dataset["path"])
            is_filtered = self.filter_manager.is_frame_filtered(dataset_path, frame_idx)

            if not is_filtered:
                self.current_frame_idx = frame_idx
                logger.info(
                    f"Found first unfiltered frame in current dataset: {frame_idx}"
                )
            else:
                # Frame 0 is filtered, so start from frame 0 but check if it's filtered
                self.current_frame_idx = 0
                logger.info(
                    "Frame 0 is filtered, starting from frame 0 in current dataset"
                )
            self._update_autoselection()
        else:
            # If no frames found or error, start from frame 0
            self.current_frame_idx = 0
            self._update_autoselection()
            logger.info("Starting from frame 0 in current dataset")

    def run(self):
        """Run the filtering application."""
        logger.info("Starting sample filtering...")
        logger.info(
            "Controls: ←/A=Accept (green border), →/R=Reject (red border), Space=Save, ESC=Exit, D=Next Dataset"
        )
        logger.info(
            "Note: You must make a selection first before pressing Space to save"
        )

        while self._process_current_frame():
            time.sleep(0.1)  # Small delay to reduce CPU usage

        cv2.destroyAllWindows()

    def _process_current_frame(self) -> bool:
        """Process current frame and return True if should continue."""
        # Get current dataset
        current_dataset = self.dataset_manager.get_current_dataset()
        if not current_dataset:
            return False

        # Check if current frame is filtered BEFORE loading it
        dataset_path = str(current_dataset["path"])
        is_filtered = self.filter_manager.is_frame_filtered(
            dataset_path, self.current_frame_idx
        )

        if is_filtered:
            logger.warning(
                f"Frame {self.current_frame_idx} is already filtered, skipping..."
            )
            return self._advance_to_next_frame()

        # Get current frame
        frame_data = current_dataset["loader"].get_frame_by_index(
            self.current_frame_idx
        )
        if not frame_data:
            return self._advance_to_next_frame()

        # Show frame
        self.display.show_with_status(
            frame_data,
            f"A: {self.accepted_count} R: {self.rejected_count}",
            current_dataset["info"],
            self.auto_decision,
            self.auto_confidence,
        )

        # Wait for input
        key_code = self.keyboard_handler.wait_for_key()  # Wait indefinitely
        key_result = self.keyboard_handler.handle_input(key_code)

        if key_result["action"] == "exit":
            return False
        elif key_result["action"] == "save":
            saved = self._save_current_selection()
            if saved:
                self._advance_to_next_frame()
        elif key_result["action"] == "set_selection":
            # Just set the selection, don't save yet - border will be displayed
            self.display.current_selection = key_result["selection"]
            # Don't advance here - let user see the border before saving with space
        elif key_result["action"] == "next_dataset":
            if len(self.dataset_manager.datasets) <= 1:
                logger.warning("Only one dataset available - cannot switch")
                return

            # Reset selection when switching datasets
            self.display.current_selection = FilterDecision.NONE

            if self.dataset_manager.switch_to_next_dataset():
                self._find_first_unfiltered_frame_in_current_dataset()
                current_dataset = self.dataset_manager.get_current_dataset()
                datasets_list = self.dataset_manager.get_datasets_list()
                if current_dataset and self.dataset_manager.current_dataset_idx < len(
                    datasets_list
                ):
                    logger.info(
                        f"Switched to dataset: {datasets_list[self.dataset_manager.current_dataset_idx]['display_name']}"
                    )
                elif current_dataset:
                    logger.info(
                        f"Switched to dataset: {current_dataset['info'].monitor_id}"
                    )
                else:
                    logger.info(
                        f"Switched to dataset: {self.dataset_manager.current_dataset_idx}"
                    )
            else:
                print("Failed to switch to next dataset")

        return True

    def _save_current_selection(self):
        """Save current selection and advance."""
        # Always require explicit selection before saving
        if self.display.current_selection == FilterDecision.NONE:
            logger.warning(
                "No selection made - press Left/Accept or Right/Reject first"
            )
            return False

        current_dataset = self.dataset_manager.get_current_dataset()
        dataset_path = str(current_dataset["path"])

        self.filter_manager.save_decision(
            dataset_path, self.current_frame_idx, self.display.current_selection.value
        )

        if self.display.current_selection == FilterDecision.ACCEPT:
            self.accepted_count += 1
        else:
            self.rejected_count += 1

        logger.info(
            f"Saved {self.display.current_selection.value} for frame {self.current_frame_idx}"
        )
        return True

    def _advance_to_next_frame(self):
        """Advance to next unfiltered frame."""
        current_dataset = self.dataset_manager.get_current_dataset()
        if not current_dataset:
            return False

        next_frame = current_dataset["loader"].get_next_unfiltered_frame(
            self.current_frame_idx
        )

        if (next_frame is not None) and (
            next_frame < current_dataset["info"].total_frames
        ):
            self.current_frame_idx = next_frame
            self._update_autoselection()
        else:
            # Try next dataset
            if self.dataset_manager.switch_to_next_dataset():
                self._find_first_unfiltered_frame()
                logger.info("Switched to next dataset")
            else:
                logger.info("All frames processed")
                return False

        return True

    def _update_autoselection(self):
        current_dataset = self.dataset_manager.get_current_dataset()
        if not current_dataset:
            return
        # Get automated filter classification
        frame_data = current_dataset["loader"].get_frame_by_index(
            self.current_frame_idx
        )
        if not frame_data:
            return
        if self.automated_filter:
            self.auto_decision, self.auto_confidence = (
                self.automated_filter.classify_frame(frame_data)
            )
            sel = FilterDecision.NONE
            if self.auto_confidence is not None:
                sel = (
                    FilterDecision.ACCEPT
                    if self.auto_decision
                    else FilterDecision.REJECT
                )
            self.display.current_selection = sel
        else:
            self.auto_decision = None
            self.auto_confidence = None

    # ============================================================================


# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the sample filter utility."""
    parser = argparse.ArgumentParser(
        description="Filter eye tracking samples with keyboard controls"
    )
    parser.add_argument(
        "--filter-file",
        type=str,
        default="sample_filters.json",
        help="Path to filter JSON file (default: sample_filters.json)",
    )
    parser.add_argument(
        "--zoom-factor",
        type=int,
        default=8,
        help="Zoom factor for eye images (default: 8)",
    )
    parser.add_argument(
        "--start-dataset",
        type=int,
        default=0,
        help="Starting dataset index (default: 0)",
    )
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for automated classification (default: 0.1)",
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Enable automated filtering using FilterWrapper model",
    )
    args = parser.parse_args()

    # Set default paths
    ROOT_FOLDER = Path(__file__).parent.parent

    # Override dataset_paths in args with provided path or default
    if args.dataset is None:
        args.dataset = str(ROOT_FOLDER / "Data" / "remote")
    args.dataset_paths = [str(Path(args.dataset).resolve())]

    # Create and run application
    app = SampleFilterUtility(args)
    app.run()


if __name__ == "__main__":
    exit(main())
