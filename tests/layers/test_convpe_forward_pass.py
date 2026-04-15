"""Tests for ConvPE forward pass."""

import tensorflow as tf

from NN.layers.EncodingLayers import ConvPE
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class TestConvPEForwardPass:
    """Tests for ConvPE forward pass."""

    def test_convpe_forward_basic(self):
        """Test ConvPE forward pass produces correct output shape."""
        layer = ConvPE(channels=16)
        layer.build((None, 32, 64, 3))

        x = tf.random.normal((4, 32, 64, 3))
        output = layer(x)

        # Test actual behavior, not just object creation
        expected_shape = (4, 32, 64, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Additional assertion to verify layer actually processes data
        # ConvPE concatenates positional encodings, so output has more channels than input
        input_channels = x.shape[-1]
        output_channels = output.shape[-1]
        assert (
            output_channels > input_channels
        ), "ConvPE should add positional encoding channels"

        # Test that the added channels contain non-zero values
        encoding_channels = output[:, :, :, input_channels:]
        assert not tf.reduce_all(
            tf.equal(encoding_channels, 0)
        ), "Positional encoding should be non-zero"

    def test_convpe_forward_with_activation(self):
        """Test ConvPE forward pass with activation produces non-negative output."""
        layer = ConvPE(channels=8, activation=STANDARD_NONNEGATIVE_ACTIVATION)
        layer.build((None, 16, 16, 3))

        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)

        expected_shape = (2, 16, 16, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: activation should produce non-negative values
        pe_channels = output[:, :, :, 3:]  # Extract PE channels
        assert (
            tf.reduce_min(pe_channels) >= 0
        ), "Activation should produce non-negative values"

    def test_convpe_forward_different_batch_sizes(self):
        """Test ConvPE forward pass works correctly with various batch sizes."""
        layer = ConvPE(channels=12)
        layer.build((None, 16, 16, 3))

        for batch_size in [1, 2, 8, 32]:
            x = tf.random.normal((batch_size, 16, 16, 3))
            output = layer(x)

            expected_shape = (batch_size, 16, 16, 3 + 12)
            assert (
                output.shape == expected_shape
            ), f"Batch {batch_size}: expected {expected_shape}, got {output.shape}"

            # Test actual behavior: output should be different from input
            # ConvPE concatenates positional encodings, so output has more channels than input
            input_channels = x.shape[-1]
            output_channels = output.shape[-1]
            assert (
                output_channels > input_channels
            ), f"ConvPE should add positional encoding channels for batch size {batch_size}"

            # Test that the added channels contain non-zero values
            encoding_channels = output[:, :, :, input_channels:]
            assert not tf.reduce_all(
                tf.equal(encoding_channels, 0)
            ), f"Positional encoding should be non-zero for batch size {batch_size}"
