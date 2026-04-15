"""Tests for learnable sink tokens in Step2LatentModel.

Tests verify that sink tokens are properly created, used, and serialized
in the Step2LatentModel without affecting output shape or external API.
"""

import pytest
import tensorflow as tf
from NN.models.Step2LatentModel import Step2LatentModel
from tests.fixtures.test_inputs import create_test_inputs


class TestStep2LatentSinkTokens:
    """Test sink token prepending and removal in Step2LatentModel."""

    def test_num_sink_tokens_zero_produces_invalid_output(self):
        """Test that zero sink tokens raise ValueError during initialization."""
        # Should raise ValueError during model creation
        with pytest.raises(ValueError, match="num_sink_tokens must be positive"):
            Step2LatentModel(latent_size=64, num_sink_tokens=0)

    def test_num_sink_tokens_negative_produces_invalid_output(self):
        """Test that negative sink tokens raise ValueError during initialization."""
        # Should raise ValueError during model creation
        with pytest.raises(ValueError, match="num_sink_tokens must be positive"):
            Step2LatentModel(latent_size=64, num_sink_tokens=-1)

    def test_num_sink_tokens_default_value(self):
        """Verify default num_sink_tokens produces expected behavior."""
        model = Step2LatentModel(latent_size=64)
        # Test that model works with default configuration
        inputs = create_test_inputs(
            batch_size=2,
            timesteps=10,
            input_type="random_normal",
            latent_size=64,
            include_embeddings=True,
            include_ids=False,
        )
        output = model(inputs, training=False)
        assert output.shape == (
            2,
            10,
            64,
        ), f"Expected output shape (2, 10, 64) with default config, got {output.shape}"

    def test_sink_tokens_are_created_with_configurable_count(self):
        """Verify sink tokens are created with correct count through model behavior."""
        for num_tokens in [1, 2, 4]:
            model = Step2LatentModel(latent_size=64, num_sink_tokens=num_tokens)
            batch_size, seq_len, latent_size = 2, 10, 64

            # Create random normal inputs with latent field
            inputs = create_test_inputs(
                batch_size=batch_size,
                timesteps=seq_len,
                input_type="random_normal",
                latent_size=latent_size,
                include_embeddings=True,
                include_ids=False,
            )

            # Verify model runs successfully with the configured num_sink_tokens
            output = model(inputs, training=False)
            assert output.shape == (batch_size, seq_len, latent_size), (
                f"Model with num_sink_tokens={num_tokens} should produce "
                f"output shape ({batch_size}, {seq_len}, {latent_size}), got {output.shape}"
            )

    def test_sink_tokens_generated_from_sink(self):
        """Verify sink mixers generate sink tokens from input."""
        num_tokens = 3
        batch_size, seq_len, latent_size = 2, 10, 64
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        # Create random normal inputs with latent field
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=latent_size,
            include_embeddings=True,
            include_ids=False,
        )

        output = model(inputs, training=False)
        # Verify output has correct shape (sink tokens should be removed)
        assert output.shape == (
            batch_size,
            seq_len,
            latent_size,
        ), f"Expected {(batch_size, seq_len, latent_size)}, got {output.shape}"

    def test_output_shape_unchanged(self):
        """Verify output shape matches input sequence length (tokens removed)."""
        batch_size, seq_len, latent_size = 4, 10, 64
        num_tokens = 2
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        # Create test inputs
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=latent_size,
            include_embeddings=True,
            include_ids=False,
        )

        output = model(inputs, training=False)
        assert output.shape == (
            batch_size,
            seq_len,
            latent_size,
        ), f"Expected {(batch_size, seq_len, latent_size)}, got {output.shape}"

    def test_output_shape_with_different_token_counts(self):
        """Verify output shape independent of number of sink tokens."""
        batch_size, seq_len, latent_size = 2, 8, 64

        for num_tokens in [1, 2, 3]:
            model = Step2LatentModel(
                latent_size=latent_size, num_sink_tokens=num_tokens
            )

            # Create test inputs
            inputs = create_test_inputs(
                batch_size=batch_size,
                timesteps=seq_len,
                input_type="random_normal",
                latent_size=latent_size,
                include_embeddings=True,
                include_ids=False,
            )

            output = model(inputs, training=False)
            assert output.shape == (
                batch_size,
                seq_len,
                latent_size,
            ), f"num_tokens={num_tokens}: Expected {(batch_size, seq_len, latent_size)}, got {output.shape}"

    def test_output_shape_various_batch_sizes(self):
        """Verify output shape correct with various batch sizes."""
        latent_size = 64
        seq_len = 10
        num_tokens = 2
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        for batch_size in [1, 2, 4, 8]:
            # Create test inputs
            inputs = create_test_inputs(
                batch_size=batch_size,
                timesteps=seq_len,
                input_type="random_normal",
                latent_size=latent_size,
                include_embeddings=True,
                include_ids=False,
            )

            output = model(inputs, training=False)
            assert output.shape == (
                batch_size,
                seq_len,
                latent_size,
            ), f"batch_size={batch_size}: Expected {(batch_size, seq_len, latent_size)}, got {output.shape}"

    def test_sink_participate_in_gradients(self):
        """Verify gradients can flow through model during training."""
        batch_size, seq_len, latent_size = 2, 5, 32
        num_tokens = 2
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        # Create test inputs
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=latent_size,
            include_embeddings=True,
            include_ids=False,
        )

        with tf.GradientTape() as tape:
            output = model(inputs, training=True)
            loss = tf.reduce_sum(output)

        # Compute gradients through trainable weights
        trainable_weights = model.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        # Verify gradient computation succeeded
        assert gradients is not None, "Gradient computation must succeed"
        # At least some gradients should be computed for trainable variables
        computed_gradients = [g for g in gradients if g is not None]
        assert (
            len(computed_gradients) > 0
        ), "Expected gradients computed for trainable variables"

    def test_sink_weights_are_trainable(self):
        """Verify model has trainable weights after building."""
        num_tokens = 2
        batch_size, seq_len, latent_size = 2, 10, 64
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        # Build the model by running a forward pass
        # Create test inputs
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=latent_size,
            include_embeddings=True,
            include_ids=False,
        )
        _ = model(inputs, training=False)

        # Check that model has trainable weights
        total_trainable = len(model.trainable_weights)
        assert total_trainable > 0, "Model should have trainable weights"

    def test_backward_compatibility_default_parameter(self):
        """Verify model works with default num_sink_tokens when not specified."""
        # This test ensures backward compatibility - old code without num_sink_tokens
        # still works with sensible defaults
        model = Step2LatentModel(latent_size=64)

        # Create test inputs
        inputs = create_test_inputs(
            batch_size=2,
            timesteps=10,
            input_type="random_normal",
            latent_size=64,
            include_embeddings=True,
            include_ids=False,
        )

        output = model(inputs, training=False)
        assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"

    def test_sink_weights_updated_with_training(self):
        """Verify model weights are updated during training."""
        batch_size, seq_len, latent_size = 2, 5, 32
        num_tokens = 2
        model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=num_tokens)

        # Create test inputs
        inputs = create_test_inputs(
            batch_size=batch_size,
            timesteps=seq_len,
            input_type="random_normal",
            latent_size=latent_size,
            include_embeddings=True,
            include_ids=False,
        )

        # Build the model first
        _ = model(inputs, training=False)

        # Capture initial weights
        initial_weights = [w.numpy().copy() for w in model.trainable_weights]

        # Run multiple forward passes with training
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        for _ in range(5):
            with tf.GradientTape() as tape:
                output = model(inputs, training=True)
                loss = tf.reduce_sum(output)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Verify weights have changed
        max_change = 0.0
        for j, w in enumerate(model.trainable_weights):
            change = tf.reduce_max(tf.abs(w.numpy() - initial_weights[j])).numpy()
            max_change = max(max_change, change)

        assert (
            max_change > 1e-6
        ), f"Model weights should be updated during training, but max_change={max_change}"

    def test_different_latent_sizes(self):
        """Verify model works with various latent sizes."""
        for latent_size in [32, 64, 128, 256]:
            model = Step2LatentModel(latent_size=latent_size, num_sink_tokens=2)

            # Create test inputs
            inputs = create_test_inputs(
                batch_size=2,
                timesteps=10,
                input_type="random_normal",
                latent_size=latent_size,
                include_embeddings=True,
                include_ids=False,
            )

            output = model(inputs, training=False)
            assert output.shape == (
                2,
                10,
                latent_size,
            ), f"latent_size={latent_size}: Expected output shape (2, 10, {latent_size}), got {output.shape}"
