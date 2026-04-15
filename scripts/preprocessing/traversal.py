"""Traversal utilities for hierarchical dataset structure.

Provides iterators and functions to traverse multi-level folder hierarchies
and access datasets at various levels (user, monitor, camera, screen, etc.).
"""

from typing import Dict, Iterator, List, Tuple, Optional
from pathlib import Path
import numpy as np
from Core.logging_config import get_logger
from Core.Constants import HIERARCHY_LEVELS

logger = get_logger(__name__)


class HierarchyTraversal:
    """Traverses hierarchical dataset folder structure.

    Supports recursive traversal of nested directories following pattern:
    rootDir/
      ├── userId1/
      │   ├── screenId1/
      │   │   ├── cameraId1/...
      │   │   └── cameraId2/...
      │   └── screenId2/...
      └── userId2/...

    Usage:
        traversal = HierarchyTraversal(root_dir)
        for user_id, user_path in traversal.iterate_level("userId"):
            # Process all data for user_id
    """

    def __init__(self, root_dir: Path):
        """Initialize traversal with root directory.

        Args:
            root_dir: Root path to dataset hierarchy.

        Raises:
            ValueError: If root_dir doesn't exist.
        """
        root_dir = Path(root_dir)

        if not root_dir.exists():
            raise ValueError(f"Root directory does not exist: {root_dir}")

        if not root_dir.is_dir():
            raise ValueError(f"Root path is not a directory: {root_dir}")

        self.root_dir = root_dir
        logger.info(f"HierarchyTraversal initialized: {root_dir}")

    def list_subdirectories(self, parent_path: Path) -> List[Path]:
        """List immediate subdirectories of given path (sorted).

        Args:
            parent_path: Path to parent directory.

        Returns:
            Sorted list of subdirectory paths.
        """
        if not parent_path.exists() or not parent_path.is_dir():
            return []

        subdirs = [p for p in parent_path.iterdir() if p.is_dir()]
        return sorted(subdirs)

    def iterate_level(
        self,
        level_name: str,
    ) -> Iterator[Tuple[str, Path]]:
        """Iterate over directories at specified level.

        For level_name="userId", yields all user directories at root level.
        For level_name="screenId", yields all screen directories within users.

        Args:
            level_name: Name of the hierarchy level (e.g., "userId", "screenId").

        Yields:
            (level_id, level_path) tuples for each directory at that level.

        Raises:
            ValueError: If level_name is not recognized.
        """
        if level_name not in HIERARCHY_LEVELS:
            raise ValueError(
                f"Unknown level '{level_name}'. Must be one of {HIERARCHY_LEVELS}"
            )

        level_idx = HIERARCHY_LEVELS.index(level_name)

        # Start with root and traverse down to target level
        # level_idx levels of traversal gives us the target level
        current_paths = [self.root_dir]

        for i in range(level_idx + 1):
            next_paths = []
            for parent in current_paths:
                subdirs = self.list_subdirectories(parent)
                next_paths.extend(subdirs)
            current_paths = next_paths

        # Yield items at target level
        for path in current_paths:
            level_id = path.name
            yield level_id, path

    def iterate_hierarchy(
        self,
        max_depth: Optional[int] = None,
    ) -> Iterator[Tuple[List[str], Path]]:
        """Recursively iterate through entire hierarchy.

        Yields paths and their hierarchy identifiers at all levels.

        Args:
            max_depth: Maximum recursion depth (None = no limit).

        Yields:
            (hierarchy_ids, path) tuples where hierarchy_ids is list of
            identifiers from root to current level.

        Example:
            For file at root/user1/screen1/camera1/file.npy:
            Yields: (["user1", "screen1", "camera1"], path)
        """

        def _traverse(current_path: Path, depth: int, ids: List[str]):
            if max_depth is not None and depth >= max_depth:
                yield ids, current_path
                return

            subdirs = self.list_subdirectories(current_path)

            if not subdirs:
                # Leaf node
                yield ids, current_path
            else:
                # Recurse into subdirectories
                for subdir in subdirs:
                    new_ids = ids + [subdir.name]
                    yield from _traverse(subdir, depth + 1, new_ids)

        yield from _traverse(self.root_dir, 0, [])

    def find_path(
        self,
        hierarchy_ids: Dict[str, str],
    ) -> Optional[Path]:
        """Find a specific path in hierarchy using level identifiers.

        Args:
            hierarchy_ids: Dictionary mapping level_name -> id.
                          Example: {"userId": "user123", "screenId": "screen1"}

        Returns:
            Path to the directory, or None if not found.
        """
        current_path = self.root_dir

        for level in HIERARCHY_LEVELS:
            if level not in hierarchy_ids:
                # We've gone as deep as needed
                return current_path

            target_name = hierarchy_ids[level]
            subdirs = self.list_subdirectories(current_path)
            found = False

            for subdir in subdirs:
                if subdir.name == target_name:
                    current_path = subdir
                    found = True
                    break

            if not found:
                logger.debug(
                    f"Could not find '{target_name}' at level '{level}' "
                    f"under {current_path}"
                )
                return None

        return current_path

    def get_depth(self, path: Path) -> int:
        """Get depth of path relative to root (0 = root itself).

        Args:
            path: Path to measure.

        Returns:
            Number of levels below root (or -1 if path not under root).
        """
        try:
            relative = path.relative_to(self.root_dir)
            # Count path segments
            return len(relative.parts)
        except ValueError:
            # Path is not relative to root_dir
            return -1


class DatasetLoader:
    """Loads datasets from storage (wrapper for future SamplesStorage integration).

    This class provides abstraction for loading datasets from various backends.
    Currently placeholder for future integration with actual storage layer.
    """

    def __init__(self, data_directory: Path):
        """Initialize loader with data directory.

        Args:
            data_directory: Root directory containing .npy files.
        """
        self.data_directory = Path(data_directory)

        if not self.data_directory.exists():
            logger.warning(f"Data directory does not exist: {data_directory}")

    def load_dataset(self, folder_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load dataset from folder.

        Loads all .npy files in folder as arrays in a dictionary.

        Args:
            folder_path: Path to folder containing .npy files.

        Returns:
            Dictionary with array names as keys, or None if no files found.

        Raises:
            FileNotFoundError: If folder_path does not exist.
            OSError: If IO error occurs reading .npy files.
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        dataset = {}
        npy_files = list(folder_path.glob("*.npy"))

        if not npy_files:
            logger.debug(f"No .npy files found in {folder_path}")
            return None

        for npy_file in npy_files:
            array_name = npy_file.stem  # Filename without .npy extension
            dataset[array_name] = np.load(npy_file)

        return dataset if dataset else None

    def save_dataset(
        self,
        dataset: Dict[str, np.ndarray],
        output_dir: Path,
        overwrite: bool = False,
    ) -> None:
        """Save dataset as .npy files.

        Args:
            dataset: Dictionary of arrays to save.
            output_dir: Directory to save files into.
            overwrite: If True, overwrite existing files.

        Raises:
            FileExistsError: If file exists and overwrite=False.
            OSError: If IO error occurs during save.
            ValueError: If dataset is empty or contains no arrays.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        array_fields = {k: v for k, v in dataset.items() if isinstance(v, np.ndarray)}
        if not array_fields:
            raise ValueError("Dataset contains no numpy arrays to save")

        for array_name, array in array_fields.items():
            output_file = output_dir / f"{array_name}.npy"

            if output_file.exists() and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {output_file}. "
                    f"Use overwrite=True to replace."
                )

            np.save(output_file, array)
            logger.debug(f"Saved {array_name} to {output_file}")

        logger.info(f"Saved {len(array_fields)} arrays to {output_dir}")


class DirectoryValidator:
    """Validates and checks directory structure."""

    @staticmethod
    def is_leaf_directory(
        path: Path, required_files: Optional[List[str]] = None
    ) -> bool:
        """Check if directory is a leaf node (no subdirectories or has required files).

        Args:
            path: Directory path to check.
            required_files: If provided, check for these file patterns.
                          Example: ["*.npy", "metadata.json"]

        Returns:
            True if directory appears to be a leaf node.
        """
        if not path.is_dir():
            return False

        # Has subdirectories -> not a leaf
        if any(p.is_dir() for p in path.iterdir()):
            return False

        # If no required_files specified, just check for absence of subdirs
        if not required_files:
            return True

        # Check for required files
        has_required = True
        for pattern in required_files:
            matching = list(path.glob(pattern))
            if not matching:
                has_required = False
                break

        return has_required

    @staticmethod
    def validate_hierarchy(root_dir: Path) -> Dict[str, int]:
        """Validate hierarchy structure and return statistics.

        Args:
            root_dir: Root directory of hierarchy.

        Returns:
            Dictionary with counts of directories at each level.

        Example:
            {"depth_1": 5, "depth_2": 20, "depth_3": 100}
        """
        stats = {}

        def _count_at_depth(path: Path, depth: int):
            if depth not in stats:
                stats[depth] = 0
            stats[depth] += 1

            subdirs = [p for p in path.iterdir() if p.is_dir()]
            for subdir in subdirs:
                _count_at_depth(subdir, depth + 1)

        _count_at_depth(root_dir, 0)
        return {f"depth_{k}": v for k, v in stats.items()}
