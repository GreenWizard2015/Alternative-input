"""Preprocessing module for dataset refactoring.

This package provides modular, testable components for preprocessing eye-tracking
datasets in hierarchical folder structures. It replaces the monolithic functions
in preprocess-remote.py with composable, well-tested utilities.

Components:
- core: Pure functions for data manipulation (expand_indices, filter deltas, etc.)
- validation: Dataset validation and sanity checking
- dataset: High-level DatasetPreprocessor orchestrator
- traversal: Hierarchical folder traversal utilities
"""

__version__ = "1.0.0"
__all__ = [
    "core",
    "validation",
    "dataset",
    "traversal",
]
