"""Unit tests for Step2LatentModel scale_mult parameter.

Tests the scale_mult scaling implementation for:
- num_heads and dff_multiplier scaling
"""

import pytest
from NN.models.Step2LatentModel import Step2LatentModel


class TestStep2LatentModelScaleMult:
    """Test suite for Step2LatentModel scale_mult parameter."""

    def test_scale_mult_default(self):
        """Test that scale_mult=1.0 is the default."""
        model = Step2LatentModel(latent_size=256)
        assert hasattr(model, "_scale_mult"), "Model should have _scale_mult attribute"
        assert (
            model._scale_mult == 1.0
        ), f"Expected _scale_mult=1.0, got {model._scale_mult}"

    def test_scale_mult_identity(self):
        """Test scale_mult=1.0 produces base dimensions."""
        model = Step2LatentModel(latent_size=256, scale_mult=1.0)

        # scaled_latent_size should be 256
        assert (
            model._scaled_latent_size == 256
        ), f"Expected scaled_latent_size=256, got {model._scaled_latent_size}"
        # num_heads computed from 16 * scale_mult = 16 * 1.0 = 16
        assert model._num_heads == 16, f"Expected num_heads=16, got {model._num_heads}"
        # dff_multiplier computed from 16 * scale_mult = 16 * 1.0 = 16
        assert (
            model._dff_multiplier == 16
        ), f"Expected dff_multiplier=16, got {model._dff_multiplier}"

    def test_scale_mult_double(self):
        """Test scale_mult=2.0 doubles dimensions."""
        model = Step2LatentModel(latent_size=256, scale_mult=2.0)

        # scaled_latent_size = 256 * 2 = 512
        assert (
            model._scaled_latent_size == 512
        ), f"Expected scaled_latent_size=512, got {model._scaled_latent_size}"
        # num_heads computed from 16 * scale_mult = 16 * 2 = 32
        assert model._num_heads == 32, f"Expected num_heads=32, got {model._num_heads}"
        # dff_multiplier computed from 16 * scale_mult = 16 * 2 = 32
        assert (
            model._dff_multiplier == 32
        ), f"Expected dff_multiplier=32, got {model._dff_multiplier}"

    def test_scale_mult_half(self):
        """Test scale_mult=0.5 halves dimensions."""
        model = Step2LatentModel(latent_size=256, scale_mult=0.5)

        # scaled_latent_size = 256 * 0.5 = 128
        assert (
            model._scaled_latent_size == 128
        ), f"Expected scaled_latent_size=128, got {model._scaled_latent_size}"
        # num_heads computed from 16 * scale_mult = 16 * 0.5 = 8
        assert model._num_heads == 8, f"Expected num_heads=8, got {model._num_heads}"
        # dff_multiplier computed from 16 * scale_mult = 16 * 0.5 = 8
        assert (
            model._dff_multiplier == 8
        ), f"Expected dff_multiplier=8, got {model._dff_multiplier}"

    def test_dff_multiplier_default_none(self):
        """Test that dff_multiplier=None is the default and gets scaled."""
        model = Step2LatentModel(latent_size=256, scale_mult=2.0, dff_multiplier=None)

        # Should compute from scale_mult: int(16 * 2.0) = 32
        assert (
            model._dff_multiplier == 32
        ), f"Expected dff_multiplier=32, got {model._dff_multiplier}"

    def test_dff_multiplier_explicit_no_scaling(self):
        """Test that explicit dff_multiplier is NOT re-scaled."""
        model = Step2LatentModel(latent_size=256, scale_mult=2.0, dff_multiplier=16)

        # Should use explicit value, not scale it
        assert (
            model._dff_multiplier == 16
        ), f"Expected dff_multiplier=16, got {model._dff_multiplier}"

    def test_num_heads_explicit_no_scaling(self):
        """Test that explicit num_heads is NOT re-scaled."""
        model = Step2LatentModel(latent_size=256, scale_mult=2.0, num_heads=16)

        # Should use explicit value, not scale it
        assert model._num_heads == 16, f"Expected num_heads=16, got {model._num_heads}"

    def test_num_heads_default_divisibility(self):
        """Test that default num_heads divides scaled_latent_size."""
        # latent_size=256, scale_mult=2.0 → scaled=512
        # target num_heads = 32, which divides 512 ✓
        model = Step2LatentModel(latent_size=256, scale_mult=2.0)

        assert (
            model._scaled_latent_size % model._num_heads == 0
        ), f"scaled_latent_size {model._scaled_latent_size} should be divisible by num_heads {model._num_heads}"

    def test_num_heads_explicit_validation(self):
        """Test that explicit num_heads divisibility is validated."""
        # latent_size=256, scale_mult=2.0 → scaled=512
        # num_heads=17 does NOT divide 512
        with pytest.raises(ValueError, match="must be divisible"):
            Step2LatentModel(latent_size=256, scale_mult=2.0, num_heads=17)

    def test_scale_mult_invalid_zero(self):
        """Test that scale_mult=0 raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            Step2LatentModel(latent_size=256, scale_mult=0.0)

    def test_scale_mult_invalid_negative(self):
        """Test that negative scale_mult raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            Step2LatentModel(latent_size=256, scale_mult=-1.0)

    def test_scale_mult_invalid_too_large(self):
        """Test that scale_mult > 10.0 succeeds but produces large values."""
        model = Step2LatentModel(latent_size=256, scale_mult=11.0)
        # Function should succeed but produce large scaled_latent_size
        assert model._scaled_latent_size == 2816  # 256 * 11
        assert model._scaled_latent_size > 1000  # Large but valid

    def test_scaled_latent_size_memory_validation(self):
        """Test that scaled_latent_size can exceed 4096 without raising ValueError."""
        # 512 * 9 = 4608 > 4096 - should now work without error
        model = Step2LatentModel(latent_size=512, scale_mult=9.0)
        assert model._scaled_latent_size == 4608

    def test_transformer_blocks_use_scaled_latent(self):
        """Test that transformer blocks are created with scaled_latent_size."""
        model = Step2LatentModel(latent_size=256, scale_mult=2.0)

        # Get first transformer block
        transformer = model.transformer_blocks[0]
        # d_model should be scaled_latent_size = 512
        assert (
            transformer.d_model == 512
        ), f"Expected d_model=512, got {transformer.d_model}"
