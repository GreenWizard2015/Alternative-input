"""Test model architecture and parameter validation (FAST PROFILE: <0.1s).

Tests mathematical validation without neural network instantiation.
Expected improvement: 150x speedup over real model validation.
"""

from tests.model_wrapper.conftest import PerformanceProfile
from Core.models.ModelWrapperProxies import ModelWrapperProxies


class TestCategory:
    """Base class for categorized performance tests."""

    PERFORMANCE_PROFILE = None
    EXPECTED_EXECUTION_TIME = None
    MEMORY_BUDGET_MB = None
    CONFIG_OPTIMIZATION = "medium"

    def get_performance_metadata(self):
        """Get metadata about test performance characteristics."""
        return {
            "profile": self.PERFORMANCE_PROFILE,
            "expected_time_s": self.EXPECTED_EXECUTION_TIME,
            "memory_budget_mb": self.MEMORY_BUDGET_MB,
            "config_optimization": self.CONFIG_OPTIMIZATION,
            "description": self.__doc__ or "",
        }


class TestArchitectureValidation(TestCategory):
    """Test model architecture and parameter validation (FAST PROFILE: <0.1s).

    Tests mathematical validation without neural network instantiation.
    Expected improvement: 150x speedup over real model validation.
    """

    PERFORMANCE_PROFILE = PerformanceProfile.ULTRA_FAST
    EXPECTED_EXECUTION_TIME = 0.1
    MEMORY_BUDGET_MB = 10
    CONFIG_OPTIMIZATION = "fast"

    def test_layer_size_validation_mathematical(self, stats_data):
        """Test layer size compatibility without model instantiation."""
        # Use proxy validation for architecture checking
        result = ModelWrapperProxies.validate_layer_sizes(
            timesteps=3,
            stats=stats_data,
            embedding_size=16,
            latent_size=128,
            mode="full",
        )

        assert result[
            "validation_passed"
        ], f"Layer size validation should pass: {result.get('warnings', [])}"
        # Check for correct keys returned by proxy validation
        assert "embeddings_vocab" in result, "Should have embeddings_vocab"
        assert "expected_latent_shape" in result, "Should have expected_latent_shape"
        assert "predictor_output_shape" in result, "Should have predictor_output_shape"
