"""Tests for main script imports and basic functions."""

import pytest
import importlib.util


@pytest.fixture
def preprocess_module():
    """Load the refactored preprocess-remote.py script."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_remote", "scripts/preprocess-remote.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMainScriptImports:
    """Test that main script imports and basic functions work."""

    def test_can_import_modules(self, preprocess_module):
        """Test that script can import new preprocessing modules."""
        # If we got here, imports succeeded
        assert hasattr(
            preprocess_module, "dropZeroTimeDelta"
        ), "Module should have dropZeroTimeDelta"
        assert hasattr(
            preprocess_module, "expand_indices"
        ), "Module should have expand_indices"
        assert hasattr(
            preprocess_module, "processFolder"
        ), "Module should have processFolder"
        assert hasattr(preprocess_module, "main"), "Module should have main"

    def test_constants_available(self, preprocess_module):
        """Test that constants are imported and accessible."""
        # The script should have imported from Constants
        # We can verify by checking that functions use them
        assert hasattr(
            preprocess_module, "MAX_DELTA_THRESHOLD"
        ), "Module should have MAX_DELTA_THRESHOLD constant"

    def test_functions_callable(self, preprocess_module):
        """Test that all expected functions are callable."""
        assert callable(
            preprocess_module.dropZeroTimeDelta
        ), "dropZeroTimeDelta should be callable"
        assert callable(
            preprocess_module.expand_indices
        ), "expand_indices should be callable"
        assert callable(
            preprocess_module.create_filtered_dataset
        ), "create_filtered_dataset should be callable"
        assert callable(
            preprocess_module.processFolder
        ), "processFolder should be callable"
        assert callable(preprocess_module.foldersList), "foldersList should be callable"
        assert callable(
            preprocess_module.traverse_hierarchy
        ), "traverse_hierarchy should be callable"
        assert callable(preprocess_module.main), "main should be callable"
