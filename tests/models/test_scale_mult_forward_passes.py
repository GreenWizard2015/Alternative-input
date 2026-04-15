"""Unit tests for model forward passes with scale_mult parameter.

Tests building and forward pass behavior with various scale_mult values.
"""

from NN.models.Step2LatentModel import Step2LatentModel
from NN.models.Face2StepModel import Face2StepModel
from NN.models.GazePredictionModel import GazePredictionModel
from tests.fixtures.test_inputs import create_test_inputs


class TestModelBuildAndForward:
    """Test suite for model building and forward passes with scale_mult."""

    def test_face2step_build_forward_scale_mult(self):
        """Test Face2StepModel build and forward pass with scale_mult."""
        model = Face2StepModel(latent_size=64, scale_mult=2.0)

        batch_size = 2
        seq_len = 5

        # Create random normal inputs with embeddings
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            include_embeddings=True,
            include_ids=False,
        )

        # Forward pass should not raise
        output = model(inputs, training=False)

        # Output shape should have scaled_latent_size
        assert output.shape == (
            batch_size,
            seq_len,
            128,
        ), f"Expected output shape ({batch_size}, {seq_len}, 128), got {output.shape}"

    def test_step2latent_forward_pass_scale_mult(self):
        """Test Step2LatentModel forward pass with scale_mult."""
        model = Step2LatentModel(latent_size=64, scale_mult=2.0)

        batch_size = 2
        seq_len = 5
        scaled_latent_size = 128  # 64 * 2.0

        # Create random normal inputs with latent field
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=scaled_latent_size,
            include_embeddings=True,
            include_ids=False,
        )

        # Forward pass should not raise
        output = model(inputs, training=False)

        # Output shape should match scaled_latent_size
        assert output.shape == (
            batch_size,
            seq_len,
            scaled_latent_size,
        ), f"Expected output shape ({batch_size}, {seq_len}, {scaled_latent_size}), got {output.shape}"

    def test_gaze_prediction_model_forward_scale_mult(self):
        """Test GazePredictionModel forward pass with scale_mult."""
        model = GazePredictionModel(latent_size=64, scale_mult=2.0)

        batch_size = 2
        seq_len = 5
        scaled_latent_size = 128  # 64 * 2.0

        # Create random normal inputs with embeddings
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            include_embeddings=True,
            include_ids=False,
        )

        # Forward pass should not raise
        output = model(inputs, training=False)

        # Should have both intermediate and final latent
        assert (
            "intermediate_latent" in output
        ), "Output should contain 'intermediate_latent' key"
        assert "final_latent" in output, "Output should contain 'final_latent' key"

        # Both should have scaled_latent_size dimension
        assert output["intermediate_latent"].shape == (
            batch_size,
            seq_len,
            scaled_latent_size,
        ), f"Expected intermediate_latent shape ({batch_size}, {seq_len}, {scaled_latent_size}), got {output['intermediate_latent'].shape}"
        assert output["final_latent"].shape == (
            batch_size,
            seq_len,
            scaled_latent_size,
        ), f"Expected final_latent shape ({batch_size}, {seq_len}, {scaled_latent_size}), got {output['final_latent'].shape}"
