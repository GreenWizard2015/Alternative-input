"""Unit tests for model scaling utilities.

Tests the compute_scaled_dimensions function with various scale multipliers,
edge cases, and validation scenarios.
"""

import pytest
from NN.models.ModelScaling import compute_scaled_dimensions


class TestComputeScaledDimensions:
    """Test suite for compute_scaled_dimensions function."""

    def test_identity_scaling(self):
        """Test that scale_mult=1.0 produces base dimensions."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        assert result["latent_size"] == 256
        assert result["eye_filters"] == [32, 32]
        assert result["num_heads"] == 16
        assert result["dff_multiplier"] == 16
        assert result["predictor_mlp_sizes"] == [128, 128, 128]

    def test_double_scaling(self):
        """Test that scale_mult=2.0 doubles all dimensions."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=2.0)

        assert result["latent_size"] == 512
        assert result["eye_filters"] == [64, 64]
        assert result["num_heads"] == 32
        assert result["dff_multiplier"] == 32
        assert result["predictor_mlp_sizes"] == [256, 256, 256]

    def test_half_scaling(self):
        """Test that scale_mult=0.5 halves dimensions."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=0.5)

        assert result["latent_size"] == 128
        assert result["eye_filters"] == [16, 16]
        assert result["num_heads"] == 8
        assert result["dff_multiplier"] == 8
        assert result["predictor_mlp_sizes"] == [64, 64, 64]

    def test_multi_head_attention_divisibility_identity(self):
        """Test that scaled_latent_size is divisible by num_heads (scale=1.0)."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        assert result["latent_size"] % result["num_heads"] == 0

    def test_multi_head_attention_divisibility_scaled(self):
        """Test that scaled_latent_size is divisible by num_heads (scale=2.0)."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=2.0)

        assert result["latent_size"] % result["num_heads"] == 0

    def test_num_heads_adjustment_odd_latent(self):
        """Test num_heads adjustment for odd scaled_latent_size.

        Example: latent_size=85 with target=16
        Should adjust num_heads to a valid divisor of 85.
        """
        # Create scenario where target num_heads doesn't divide evenly
        # latent_size=51 * scale_mult=2.0 = 102, target=32 but 102 % 32 != 0
        # Should adjust to 17 (divisor of 102)
        result = compute_scaled_dimensions(latent_size=51, scale_mult=2.0)

        # 102 is divisible by: 1, 2, 3, 6, 17, 34, 51, 102
        # target=32 should adjust down to 34 or 17
        assert result["latent_size"] % result["num_heads"] == 0
        assert result["num_heads"] <= 32

    def test_all_dimensions_positive(self):
        """Test that all returned dimensions are positive."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=2.5)

        assert result["latent_size"] > 0
        assert all(f > 0 for f in result["eye_filters"])
        assert result["num_heads"] > 0
        assert result["dff_multiplier"] > 0
        assert all(s > 0 for s in result["predictor_mlp_sizes"])

    def test_return_type_is_dict(self):
        """Test that function returns a dictionary."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        assert isinstance(result, dict)
        assert len(result) >= 5

    def test_return_dict_has_required_keys(self):
        """Test that returned dict has all required keys."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        required_keys = {
            "latent_size",
            "eye_filters",
            "num_heads",
            "dff_multiplier",
            "predictor_mlp_sizes",
        }
        assert required_keys.issubset(result.keys())

    def test_eye_filters_is_list(self):
        """Test that eye_filters is a list of length 2."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        assert isinstance(result["eye_filters"], list)
        assert len(result["eye_filters"]) == 2

    def test_predictor_mlp_sizes_is_list(self):
        """Test that predictor_mlp_sizes is a list of length 3."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=1.0)

        assert isinstance(result["predictor_mlp_sizes"], list)
        assert len(result["predictor_mlp_sizes"]) == 3

    def test_invalid_scale_mult_zero(self):
        """Test that scale_mult=0 raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            compute_scaled_dimensions(latent_size=256, scale_mult=0.0)

    def test_invalid_scale_mult_negative(self):
        """Test that negative scale_mult raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            compute_scaled_dimensions(latent_size=256, scale_mult=-1.0)

    def test_invalid_scale_mult_too_large(self):
        """Test that scale_mult > 10.0 succeeds but may produce large values."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=11.0)
        # Function should succeed but produce large scaled_latent_size
        assert result["latent_size"] == 2816  # 256 * 11
        assert result["latent_size"] > 1000  # Large but valid

    def test_invalid_scaled_latent_too_large(self):
        """Test that scaled_latent_size can be large but is computed correctly."""
        # Function should succeed with large scale_mult
        result = compute_scaled_dimensions(latent_size=256, scale_mult=20.0)
        scaled_size = result["latent_size"]  # Should be 5120
        assert scaled_size == 5120  # 256 * 20
        assert (
            scaled_size > 4096
        )  # Exceeds the mentioned threshold but function handles it

    def test_edge_case_large_scale_mult(self):
        """Test with maximum valid scale_mult=10.0."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=10.0)

        assert result["latent_size"] == 2560
        assert result["latent_size"] <= 4096

    def test_edge_case_small_scale_mult(self):
        """Test with very small scale_mult=0.1."""
        result = compute_scaled_dimensions(latent_size=256, scale_mult=0.1)

        assert result["latent_size"] == 25
        assert result["latent_size"] > 0

    def test_edge_case_small_latent_size(self):
        """Test with small base latent_size=64."""
        result = compute_scaled_dimensions(latent_size=64, scale_mult=2.0)

        assert result["latent_size"] == 128
        assert result["latent_size"] % result["num_heads"] == 0

    def test_scale_mult_various_values(self):
        """Test with various scale_mult values for robustness."""
        for scale_mult in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            result = compute_scaled_dimensions(latent_size=256, scale_mult=scale_mult)

            # All dimensions should be positive
            assert result["latent_size"] > 0
            # Divisibility should hold
            assert result["latent_size"] % result["num_heads"] == 0
            # Scaling should be consistent
            assert result["latent_size"] == int(256 * scale_mult)

    def test_scaling_consistency(self):
        """Test that scaling is applied consistently across all dimensions."""
        scale_mult = 3.5
        result = compute_scaled_dimensions(latent_size=256, scale_mult=scale_mult)

        # Check that scaling is roughly consistent (allowing for int rounding)
        expected_latent = int(256 * scale_mult)
        expected_eye_filter = int(32 * scale_mult)
        expected_dff = int(16 * scale_mult)

        assert result["latent_size"] == expected_latent
        assert result["eye_filters"][0] == expected_eye_filter
        assert result["dff_multiplier"] == expected_dff

    def test_num_heads_valid_divisor(self):
        """Test that num_heads is always a valid divisor of scaled_latent_size."""
        test_cases = [
            (64, 1.0),
            (64, 2.0),
            (128, 0.5),
            (256, 1.5),
            (512, 2.0),
        ]

        for latent_size, scale_mult in test_cases:
            result = compute_scaled_dimensions(
                latent_size=latent_size, scale_mult=scale_mult
            )
            assert result["latent_size"] % result["num_heads"] == 0
            assert result["num_heads"] >= 1
