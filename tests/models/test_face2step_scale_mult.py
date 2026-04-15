"""Unit tests for Face2StepModel scale_mult parameter.

Tests the scale_mult scaling implementation for MLP and Dense layers.
"""

from NN.models.Face2StepModel import Face2StepModel


class TestFace2StepModelScaleMult:
    """Test suite for Face2StepModel scale_mult parameter."""

    def test_scale_mult_default(self):
        """Test that scale_mult=1.0 is the default."""
        model = Face2StepModel(latent_size=256)
        assert (
            model._scale_mult == 1.0
        ), f"Expected _scale_mult=1.0, got {model._scale_mult}"

    def test_scale_mult_identity(self):
        """Test scale_mult=1.0 produces base dimensions."""
        model = Face2StepModel(latent_size=256, scale_mult=1.0)

        # scaled_latent_size should be 256
        assert (
            model._scaled_latent_size == 256
        ), f"Expected scaled_latent_size=256, got {model._scaled_latent_size}"

    def test_scale_mult_double(self):
        """Test scale_mult=2.0 doubles scaled_latent_size."""
        model = Face2StepModel(latent_size=256, scale_mult=2.0)

        # scaled_latent_size = 256 * 2 = 512
        assert (
            model._scaled_latent_size == 512
        ), f"Expected scaled_latent_size=512, got {model._scaled_latent_size}"
