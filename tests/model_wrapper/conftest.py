"""Configuration and fixtures for model_wrapper tests."""

# Import from centralized location for backward compatibility
from tests.fixtures.test_inputs import create_test_inputs

__all__ = ["create_test_inputs", "PerformanceProfile"]


# Performance constants for test categorization
class PerformanceProfile:
    """Performance profiles for different test categories."""

    # Ultra-fast tests (< 0.1s) - Proxy validation only
    ULTRA_FAST = "ultra_fast"

    # Fast tests (< 1s) - Mock models with minimal configurations
    FAST = "fast"

    # Medium tests (< 5s) - Real models with optimized configurations
    MEDIUM = "medium"

    # Slow tests (< 10s) - Real models with full configurations
    SLOW = "slow"

    # Integration tests (< 30s) - Full end-to-end validation
    INTEGRATION = "integration"


def pytest_addoption(parser):
    """Add command line options for performance-based test selection."""
    group = parser.getgroup("performance")
    group.addoption(
        "--profile",
        action="store",
        choices=["ultra_fast", "fast", "medium", "slow", "integration"],
        help="Run tests only from specific performance profile",
    )
    group.addoption(
        "--skip-slow", action="store_true", help="Skip slow tests (integration tests)"
    )
    group.addoption(
        "--skip-integration", action="store_true", help="Skip integration tests"
    )
