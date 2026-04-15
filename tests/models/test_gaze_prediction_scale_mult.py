"""Unit tests for GazePredictionModel scale_mult integration.

Tests the scale_mult parameter propagation to sub-models.
"""

import pytest
from NN.models.GazePredictionModel import GazePredictionModel


class TestGazePredictionModelScaleMult:
    """Test suite for GazePredictionModel scale_mult integration."""

    def test_scale_mult_default(self):
        """Test that scale_mult=1.0 is the default."""
        model = GazePredictionModel(latent_size=64)
        assert (
            model._scale_mult == 1.0
        ), f"Expected _scale_mult=1.0, got {model._scale_mult}"

    def test_scale_mult_passed_to_sub_models(self):
        """Test that scale_mult is passed to Face2Step and Step2Latent."""
        model = GazePredictionModel(latent_size=64, scale_mult=2.0)

        # Check that sub-models received scale_mult
        assert (
            model.face2step._scale_mult == 2.0
        ), f"Expected face2step._scale_mult=2.0, got {model.face2step._scale_mult}"
        assert (
            model.step2latent._scale_mult == 2.0
        ), f"Expected step2latent._scale_mult=2.0, got {model.step2latent._scale_mult}"

    def test_sub_models_compute_scaled_dimensions(self):
        """Test that sub-models independently compute scaled dimensions."""
        model = GazePredictionModel(latent_size=64, scale_mult=2.0)

        # Base latent_size is passed to sub-models
        assert (
            model.face2step.latent_size == 64
        ), f"Expected face2step.latent_size=64, got {model.face2step.latent_size}"
        assert (
            model.step2latent.latent_size == 64
        ), f"Expected step2latent.latent_size=64, got {model.step2latent.latent_size}"

        # But they compute scaled_latent_size locally
        assert (
            model.face2step._scaled_latent_size == 128
        ), f"Expected face2step._scaled_latent_size=128, got {model.face2step._scaled_latent_size}"
        assert (
            model.step2latent._scaled_latent_size == 128
        ), f"Expected step2latent._scaled_latent_size=128, got {model.step2latent._scaled_latent_size}"

    def test_scale_mult_multiple_values(self):
        """Test GazePredictionModel with various scale_mult values."""
        for scale_mult in [0.5, 1.0, 2.0, 3.0]:
            model = GazePredictionModel(latent_size=64, scale_mult=scale_mult)

            # Verify scaling is applied consistently
            expected_scaled = int(64 * scale_mult)
            assert (
                model.face2step._scaled_latent_size == expected_scaled
            ), f"For scale_mult={scale_mult}, expected face2step._scaled_latent_size={expected_scaled}, got {model.face2step._scaled_latent_size}"
            assert (
                model.step2latent._scaled_latent_size == expected_scaled
            ), f"For scale_mult={scale_mult}, expected step2latent._scaled_latent_size={expected_scaled}, got {model.step2latent._scaled_latent_size}"

    def test_scale_mult_invalid_zero(self):
        """Test that scale_mult=0 raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            GazePredictionModel(latent_size=64, scale_mult=0.0)

    def test_scale_mult_invalid_negative(self):
        """Test that negative scale_mult raises ValueError."""
        with pytest.raises(ValueError, match="scale_mult must be positive"):
            GazePredictionModel(latent_size=64, scale_mult=-1.0)
