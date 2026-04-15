"""Tests for ConvPE edge cases."""

import tensorflow as tf

from NN.layers.EncodingLayers import ConvPE


class TestConvPEEdgeCases:
    """Tests for ConvPE edge cases."""

    def test_convpe_minimum_spatial_size(self):
        """Test ConvPE with 1x1 spatial dimensions."""
        layer = ConvPE(channels=16)
        layer.build((None, 1, 1, 3))

        x = tf.random.normal((1, 1, 1, 3))
        output = layer(x)

        expected_shape = (1, 1, 1, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should work with minimum spatial size
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

    def test_convpe_large_spatial_size(self):
        """Test ConvPE with large spatial dimensions."""
        layer = ConvPE(channels=8)
        layer.build((None, 1024, 1024, 3))

        x = tf.random.normal((1, 1024, 1024, 3))
        output = layer(x)

        expected_shape = (1, 1024, 1024, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should handle large spatial dimensions
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

    def test_convpe_asymmetric_spatial(self):
        """Test ConvPE with asymmetric spatial dimensions."""
        layer = ConvPE(channels=12)
        layer.build((None, 64, 256, 3))

        x = tf.random.normal((2, 64, 256, 3))
        output = layer(x)

        expected_shape = (2, 64, 256, 3 + 12)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should work with asymmetric dimensions
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
