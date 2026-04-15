"""Face mesh landmark constants and processing functions.

Contains landmark connection data, color mappings, and functions for decoding
MediaPipe face mesh landmarks into point arrays.
"""

from typing import Any
import numpy as np
from collections import defaultdict

FACE_PARTS_CONNECTIONS = {
    "lips": [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ],
    "left eye": [
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        (263, 466),
        (466, 388),
        (388, 387),
        (387, 386),
        (386, 385),
        (385, 384),
        (384, 398),
        (398, 362),
    ],
    "left eyebrow": [
        (276, 283),
        (283, 282),
        (282, 295),
        (295, 285),
        (300, 293),
        (293, 334),
        (334, 296),
        (296, 336),
    ],
    "right eye": [
        (33, 7),
        (7, 163),
        (163, 144),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
    ],
    "right eyebrow": [
        (46, 53),
        (53, 52),
        (52, 65),
        (65, 55),
        (70, 63),
        (63, 105),
        (105, 66),
        (66, 107),
    ],
    "face oval": [
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    ],
}

COLORS = {
    "lips": (255, 255, 255),
    "left eye": (255, 255, 0),
    "right eye": (0, 255, 255),
    "face oval": (0, 255, 0),
    "right eyebrow": (255, 0, 0),
    "left eyebrow": (255, 0, 0),
}

# Build index mappings for landmark parts
INDEX_TO_PART = {}
PART_TO_INDICES = defaultdict(set)
for part_name, pairs in FACE_PARTS_CONNECTIONS.items():
    for pair in pairs:
        for landmark_index in pair:
            INDEX_TO_PART[landmark_index] = part_name
            PART_TO_INDICES[part_name].add(landmark_index)

# Constants for face mesh processing
FACE_MESH_INVALID_VALUE = -10.0
FACE_MESH_POINTS = 478


def decode_landmarks(
    landmarks: Any,
    visibility_threshold: float,
    presence_threshold: float,  # type: ignore[unused-argument]  # Kept for API compatibility with MediaPipe landmark decoders
) -> np.ndarray:
    """Decode face mesh landmarks to point coordinates.

    Converts MediaPipe landmarks protobuf to numpy array, filtering by visibility
    threshold. Invalid landmarks are filled with FACE_MESH_INVALID_VALUE.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList protobuf object
        visibility_threshold: Minimum visibility score (0.0-1.0) to include landmark
        presence_threshold: Minimum presence score (unused, kept for API compatibility)

    Returns:
        Array of shape (FACE_MESH_POINTS, 2) with (x, y) coordinates or
        FACE_MESH_INVALID_VALUE (-10.0) for filtered landmarks
    """
    points = np.full(
        (FACE_MESH_POINTS, 2), fill_value=FACE_MESH_INVALID_VALUE, dtype=np.float32
    )
    for idx, mark in enumerate(landmarks.landmark):
        if mark.HasField("visibility") and (mark.visibility < visibility_threshold):
            continue

        points[idx, 0] = mark.x
        points[idx, 1] = mark.y

    return points
