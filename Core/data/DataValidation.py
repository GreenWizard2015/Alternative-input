"""Data validation utilities for training data samples.

Provides validation functions to ensure data samples meet structural and
content requirements before training.
"""

from typing import Any


def validate_sample_structure(sample: Any) -> None:
    """Validate that a sample has the required structure.

    Checks that sample is a tuple with input (X) and output (Y) components,
    and that the input contains the required keys for both clean and augmented
    data variants.

    Args:
        sample: Sample tuple of (X, Y) where X is a dict with 'clean' and
            'augmented' keys, and Y is the target output.

    Raises:
        AssertionError: If sample structure is invalid (not a 2-tuple, X not dict,
            missing 'clean' or 'augmented' keys).

    Example:
        >>> sample = ({"clean": {...}, "augmented": {...}}, np.array([0.5, 0.5]))
        >>> validate_sample_structure(sample)  # Passes if structure valid
    """
    assert isinstance(sample, tuple), "Sample must be a tuple (X, Y)"
    assert (
        len(sample) == 2
    ), "Sample must have exactly 2 elements: input (X) and output (Y)"

    X, _ = sample
    assert isinstance(X, dict), "Input (X) must be a dictionary"

    # Validate presence of clean and augmented variants
    assert "clean" in X, "The input should contain the clean data"
    assert "augmented" in X, "The input should contain the augmented data"


def validate_sample_content(sample: Any) -> None:
    """Validate that sample contains all required fields.

    Checks that both clean and augmented variants contain all required
    fields: points, left eye, right eye, time, userId, placeId, screenId.

    Args:
        sample: Sample tuple of (X, Y) where X is a dict with 'clean' and
            'augmented' keys, each containing required fields.

    Raises:
        AssertionError: If required fields are missing from clean or augmented
            variants.

    Example:
        >>> sample = ({"clean": {...with all fields...}, "augmented": {...}}, y)
        >>> validate_sample_content(sample)  # Passes if all fields present
    """
    validate_sample_structure(sample)

    X, _ = sample
    required_fields = [
        "points",
        "left eye",
        "right eye",
        "time",
        "userId",
        "placeId",
        "screenId",
    ]

    for variant in ["clean", "augmented"]:
        item = X[variant]
        for field in required_fields:
            assert (
                field in item
            ), f"The input {variant} variant should contain the {field}"


def validate_sample(sample: Any) -> None:
    """Validate sample has both structure and content requirements.

    Performs comprehensive validation including structure and content checks.
    This is the main validation entry point that combines structure and content
    validation.

    Args:
        sample: Sample tuple of (X, Y) where X is a dict with 'clean' and
            'augmented' keys containing all required fields, and Y is target.

    Raises:
        AssertionError: If sample fails structure or content validation.

    Example:
        >>> sample = ({"clean": {...}, "augmented": {...}}, gaze_target)
        >>> validate_sample(sample)  # Comprehensive validation
    """
    validate_sample_structure(sample)
    validate_sample_content(sample)
