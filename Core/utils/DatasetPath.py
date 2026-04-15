"""DatasetPath class for standardized dataset path construction and extraction."""

from pathlib import Path
from typing import Optional


class DatasetPath:
    """Standardized path construction and extraction for dataset organization.

    Represents a dataset location with user, screen, camera, monitor, and place IDs.
    The dataset is organized as: userId/screenId/cameraId/monitorId/placeId/
    This represents the hierarchy: each user has multiple screens, each screen has multiple cameras,
    each camera has multiple monitors, and each monitor captures data from a unique place.

    This class centralizes all path operations with stored IDs, using pathlib.Path
    for cross-platform compatibility.

    Attributes:
        user_id: User identifier
        screen_id: Screen identifier
        camera_id: Camera identifier
        monitor_id: Monitor identifier
        place_id: Place identifier
        base_path: Optional base path for absolute path operations

    Example:
        >>> path = DatasetPath(user_id="user1", screen_id="screen1", camera_id="camera1", monitor_id="monitor1", place_id="place1")
        >>> path.full_path
        'user1/screen1/camera1/monitor1/place1'
    """

    def __init__(
        self,
        user_id: str,
        screen_id: str,
        camera_id: str,
        monitor_id: str,
        place_id: str,
        base_path: Optional[str] = None,
    ):
        """Initialize path handler with 5 IDs and base path.

        Args:
            user_id: User identifier
            screen_id: Screen identifier
            camera_id: Camera identifier
            monitor_id: Monitor identifier
            place_id: Place identifier
            base_path: Optional base path for absolute path operations (default: None)

        Raises:
            ValueError: If any ID is empty.
        """
        ids = {
            "user_id": user_id,
            "screen_id": screen_id,
            "camera_id": camera_id,
            "monitor_id": monitor_id,
            "place_id": place_id,
        }
        for id_name, id_value in ids.items():
            if not id_value:
                raise ValueError(f"{id_name} must not be empty")

        self.user_id = user_id
        self.screen_id = screen_id
        self.camera_id = camera_id
        self.monitor_id = monitor_id
        self.place_id = place_id
        self.base_path = base_path

    @property
    def full_path(self) -> str:
        """Full path for dataset tracking (userId/screenId/cameraId/monitorId/placeId).

        Returns absolute path if base_path is set; otherwise returns relative path.
        """
        path = (
            Path(self.user_id)
            / self.screen_id
            / self.camera_id
            / self.monitor_id
            / self.place_id
        )
        if self.base_path is not None:
            return str(Path(self.base_path) / path)
        return str(path)

    @classmethod
    def from_path(
        cls, file_path: str, base_path: Optional[str] = None
    ) -> "DatasetPath":
        """Create DatasetPath by extracting IDs from file or folder path.

        Expected format: .../userId/screenId/cameraId/monitorId/placeId or .../userId/screenId/cameraId/monitorId/placeId/filename

        Args:
            file_path: Path to file or folder (e.g., "/data/user1/screen1/camera1/monitor1/place1/train.npz" or "/data/user1/screen1/camera1/monitor1/place1")
            base_path: Optional base path for absolute path operations

        Returns:
            DatasetPath instance with extracted IDs from the input path.

        Raises:
            ValueError: If path does not contain at least 5 directory levels

        Example:
            >>> path = DatasetPath.from_path("/data/user1/screen1/camera1/monitor1/place1/train.npz")
            >>> path.full_path
            'user1/screen1/camera1/monitor1/place1'
            >>> path2 = DatasetPath.from_path("/data/user1/screen1/camera1/monitor1/place1")
            >>> path2.full_path
            'user1/screen1/camera1/monitor1/place1'
        """
        # Handle both file paths and folder paths
        path_obj = Path(file_path)

        # If it's an actual file, extract its directory
        if path_obj.is_file():
            folder_path = path_obj.parent
        else:
            folder_path = path_obj

        parts = folder_path.parts

        if len(parts) < 5:
            raise ValueError(
                f"Invalid path format: {file_path}. Expected userId/screenId/cameraId/monitorId/placeId/filename or userId/screenId/cameraId/monitorId/placeId"
            )

        user_id, screen_id, camera_id, monitor_id, place_id = (
            parts[-5],
            parts[-4],
            parts[-3],
            parts[-2],
            parts[-1],
        )
        return cls(user_id, screen_id, camera_id, monitor_id, place_id, base_path)
