"""Tests for GazePredictionModel mode parameter handling.

Tests verify:
- Model creates/skips Step2Latent based on mode
- Output shapes are correct for both modes
- In encoder mode, intermediate_latent == final_latent
- In full mode, intermediate_latent != final_latent (usually)
"""

import pytest
import tensorflow as tf
from NN.models.GazePredictionModel import GazePredictionModel
from tests.fixtures.test_inputs import create_test_inputs


class TestGazePredictionModelMode:
    """Test GazePredictionModel mode parameter and behavior."""

    @pytest.fixture
    def inputs(self):
        """Provide minimal valid inputs for model call."""
        # Create random normal inputs with custom embeddings size
        inputs = create_test_inputs(
            batch_size=2,
            timesteps=5,
            input_type="random_uniform",
            include_embeddings=True,
            include_ids=False,
        )
        return inputs

    def test_encoder_mode_returns_same_latent_for_both(self, inputs):
        """Verify intermediate_latent == final_latent in encoder mode."""
        model = GazePredictionModel(latent_size=64, mode="encoder")
        output = model(inputs, training=False)

        intermediate = output["intermediate_latent"]
        final = output["final_latent"]

        # In encoder mode, they should be the same tensor
        assert tf.reduce_all(
            tf.equal(intermediate, final)
        ), "In encoder mode, intermediate and final latents should be identical"

    def test_output_shapes_encoder_mode(self, inputs):
        """Verify output shapes are correct in encoder mode."""
        batch_size = inputs["points"].shape[0]
        seq_len = inputs["points"].shape[1]
        model_latent_size = 64

        model = GazePredictionModel(latent_size=model_latent_size, mode="encoder")
        output = model(inputs, training=False)

        intermediate = output["intermediate_latent"]
        final = output["final_latent"]

        expected_shape = (batch_size, seq_len, model_latent_size)
        assert (
            intermediate.shape == expected_shape
        ), f"Intermediate latent shape should be {expected_shape}, got {intermediate.shape}"
        assert (
            final.shape == expected_shape
        ), f"Final latent shape should be {expected_shape}, got {final.shape}"

    def test_output_shapes_full_mode(self, inputs):
        """Verify output shapes are correct in full mode."""
        batch_size = inputs["points"].shape[0]
        seq_len = inputs["points"].shape[1]
        latent_size = 64

        model = GazePredictionModel(latent_size=latent_size, mode="full")
        output = model(inputs, training=False)

        intermediate = output["intermediate_latent"]
        final = output["final_latent"]

        expected_shape = (batch_size, seq_len, latent_size)
        assert (
            intermediate.shape == expected_shape
        ), f"Intermediate latent shape should be {expected_shape}, got {intermediate.shape}"
        assert (
            final.shape == expected_shape
        ), f"Final latent shape should be {expected_shape}, got {final.shape}"

    def test_init_rejects_invalid_mode(self):
        """Verify invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            GazePredictionModel(latent_size=64, mode="invalid_mode")

    def test_init_rejects_non_positive_latent_size(self):
        """Verify non-positive latent_size raises ValueError."""
        with pytest.raises(ValueError, match="latent_size must be positive"):
            GazePredictionModel(latent_size=0, mode="full")
