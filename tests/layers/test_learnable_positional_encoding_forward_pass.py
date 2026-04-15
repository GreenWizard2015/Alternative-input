"""Tests for LearnablePositionalEncoding forward pass."""

import tensorflow as tf

from NN.layers.EncodingLayers import LearnablePositionalEncoding
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class TestLearnablePositionalEncodingForwardPass:
    """Tests for LearnablePositionalEncoding forward pass."""

    def test_forward_spatial_pe(self):
        """Test forward pass with spatial axes produces correct output shape."""
        layer = LearnablePositionalEncoding(channels=16, axis=[-3, -2])
        input_shape = (4, 32, 64, 3)

        # Build layer
        layer.build((None, 32, 64, 3))

        # Create input and test actual behavior
        x = tf.random.normal(input_shape)
        output = layer(x)

        # Test actual output shape
        expected_shape = (4, 32, 64, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test that layer actually processes data by checking PE channels are not zero
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_width_axis_only(self):
        """Test forward pass with width axis only."""
        layer = LearnablePositionalEncoding(channels=8, axis=[-2])
        input_shape = (2, 16, 32, 3)

        layer.build((None, 16, 32, 3))
        x = tf.random.normal(input_shape)
        output = layer(x)

        expected_shape = (2, 16, 32, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_single_batch(self):
        """Test forward pass with batch size 1."""
        layer = LearnablePositionalEncoding(channels=32, axis=[-3, -2])
        input_shape = (1, 8, 8, 3)

        layer.build((None, 8, 8, 3))
        x = tf.random.normal(input_shape)
        output = layer(x)

        expected_shape = (1, 8, 8, 3 + 32)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels even with single batch
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        layer = LearnablePositionalEncoding(channels=16, axis=[-3, -2])
        input_shape = (128, 32, 64, 3)

        layer.build((None, 32, 64, 3))
        x = tf.random.normal(input_shape)
        output = layer(x)

        expected_shape = (128, 32, 64, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_different_input_shapes(self):
        """Test forward pass works with different spatial dimensions."""
        layer = LearnablePositionalEncoding(channels=8)

        # Test with 16x16
        layer.build((None, 16, 16, 3))
        x1 = tf.random.normal((2, 16, 16, 3))
        output1 = layer(x1)
        expected_shape1 = (2, 16, 16, 3 + 8)
        assert (
            output1.shape == expected_shape1
        ), f"Expected shape {expected_shape1} for 16x16, got {output1.shape}"

        # Test actual behavior: should create non-zero PE channels
        pe_channels1 = output1[
            :, :, :, x1.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels1, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

        # Create new layer for different size
        layer2 = LearnablePositionalEncoding(channels=8)
        layer2.build((None, 32, 32, 3))
        x2 = tf.random.normal((2, 32, 32, 3))
        output2 = layer2(x2)
        expected_shape2 = (2, 32, 32, 3 + 8)
        assert (
            output2.shape == expected_shape2
        ), f"Expected shape {expected_shape2} for 32x32, got {output2.shape}"

        # Test actual behavior: should create non-zero PE channels
        pe_channels2 = output2[
            :, :, :, x2.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels2, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_preserves_input_values(self):
        """Test that input values appear unchanged in output while PE is added."""
        layer = LearnablePositionalEncoding(channels=4)
        layer.build((None, 8, 8, 3))

        x = tf.constant(
            [[[1.0, 2.0, 3.0] for _ in range(8)] for _ in range(8)], dtype=tf.float32
        )
        x = tf.expand_dims(x, axis=0)

        output = layer(x)

        # First 3 channels should match input
        output_input_channels = output[:, :, :, :3]
        tf.debugging.assert_near(
            output_input_channels,
            x,
            message="Input channels should be preserved in output",
        )

        # Test actual behavior: PE channels should be non-zero
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "PE channels should be non-zero"

    def test_forward_pe_values_are_different(self):
        """Test that PE channels contain different values than zeros."""
        layer = LearnablePositionalEncoding(channels=8)
        layer.build((None, 8, 8, 3))

        x = tf.zeros((1, 8, 8, 3))
        output = layer(x)

        # PE channels (last 8) should not be all zeros
        pe_channels = output[:, :, :, 3:]
        pe_sum = tf.reduce_sum(tf.abs(pe_channels))

        assert pe_sum > 0.0, "PE channels should contain non-zero values"

        # Test actual behavior: PE channels should be non-zero
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "PE channels should be non-zero"

    def test_forward_with_activation(self):
        """Test forward pass applies activation to PE and produces correct output."""
        layer = LearnablePositionalEncoding(
            channels=8, activation=STANDARD_NONNEGATIVE_ACTIVATION
        )
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((1, 8, 8, 3))
        output = layer(x)

        # Should produce correct shape with activation applied
        expected_shape = (1, 8, 8, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test activation behavior: PE channels should be non-negative
        pe_channels = output[:, :, :, 3:]
        assert (
            tf.reduce_min(pe_channels) >= 0
        ), "Activation should produce non-negative values"

        # Test actual behavior: should create non-zero PE channels
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_forward_consistency(self):
        """Test that forward pass is deterministic (same input, same PE in output)."""
        layer = LearnablePositionalEncoding(channels=8)
        layer.build((None, 8, 8, 3))

        x1 = tf.ones((1, 8, 8, 3))
        x2 = tf.ones((1, 8, 8, 3))

        output1 = layer(x1)
        output2 = layer(x2)

        # PE channels should be identical for same input
        pe1 = output1[:, :, :, 3:]
        pe2 = output2[:, :, :, 3:]

        tf.debugging.assert_near(pe1, pe2, message="PE should be deterministic")

        # Test that input is preserved
        tf.debugging.assert_near(
            output1[:, :, :, :3],
            output2[:, :, :, :3],
            message="Input channels should be identical",
        )
