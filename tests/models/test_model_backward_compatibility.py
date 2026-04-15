"""Unit tests for backward compatibility with scale_mult=1.0.

Tests that models with scale_mult=1.0 match legacy behavior.
"""

from NN.models.Step2LatentModel import Step2LatentModel
from NN.models.EyeEncoder import EyeEncoder
from NN.models.Face2StepModel import Face2StepModel


class TestBackwardCompatibility:
    """Test suite for backward compatibility with scale_mult=1.0."""

    def test_step2latent_backward_compat(self):
        """Test Step2LatentModel with scale_mult=1.0 matches old behavior."""
        # Create models with explicit scale_mult=1.0 and default
        model_explicit = Step2LatentModel(latent_size=256, scale_mult=1.0)
        model_default = Step2LatentModel(latent_size=256)

        assert (
            model_explicit._scaled_latent_size == model_default._scaled_latent_size
        ), "Explicit scale_mult=1.0 should match default"
        assert (
            model_explicit._num_heads == model_default._num_heads
        ), "num_heads should match between explicit and default"
        assert (
            model_explicit._dff_multiplier == model_default._dff_multiplier
        ), "dff_multiplier should match between explicit and default"

    def test_face2step_backward_compat(self):
        """Test Face2StepModel with scale_mult=1.0 matches old behavior."""
        model_explicit = Face2StepModel(latent_size=256, scale_mult=1.0)
        model_default = Face2StepModel(latent_size=256)

        assert (
            model_explicit._scaled_latent_size == model_default._scaled_latent_size
        ), "Explicit scale_mult=1.0 should match default for Face2StepModel"

    def test_eye_encoder_backward_compat(self):
        """Test EyeEncoder with scale_mult=1.0 matches old behavior."""
        encoder_explicit = EyeEncoder(latent_size=256, scale_mult=1.0)
        encoder_default = EyeEncoder(latent_size=256)

        assert (
            encoder_explicit._scale_mult == encoder_default._scale_mult
        ), "Explicit scale_mult=1.0 should match default for EyeEncoder"
