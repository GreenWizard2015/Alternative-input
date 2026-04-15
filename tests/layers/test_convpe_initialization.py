"""Tests for ConvPE shortcut initialization."""

import tensorflow as tf

from NN.layers.EncodingLayers import ConvPE
from NN.Constants import STANDARD_NONNEGATIVE_ACTIVATION


class TestConvPEInitialization:
    """Tests for ConvPE shortcut initialization."""

    def test_convpe_default_channels(self):
        """Test ConvPE uses default channels and produces correct output."""
        layer = ConvPE()
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((1, 8, 8, 3))
        output = layer(x)

        # Test actual behavior, not just object creation
        expected_channels = 3 + 32  # default channels = 32
        assert (
            output.shape[-1] == expected_channels
        ), f"Expected {expected_channels} output channels, got {output.shape[-1]}"

        # Test that layer actually processes data by checking PE channels are not zero
        pe_channels = output[:, :, :, 3:]  # Get the PE channels (after first 3)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "ConvPE should create non-zero PE channels"

    def test_convpe_custom_channels(self):
        """Test ConvPE accepts custom channels and applies them correctly."""
        layer = ConvPE(channels=48)
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((1, 8, 8, 3))
        output = layer(x)

        expected_channels = 3 + 48
        assert (
            output.shape[-1] == expected_channels
        ), f"Expected {expected_channels} output channels, got {output.shape[-1]}"

        # Test that layer actually processes data by checking PE channels are not zero
        pe_channels = output[:, :, :, 3:]  # Get the PE channels (after first 3)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "ConvPE should create non-zero PE channels"

    def test_convpe_fixed_axes(self):
        """Test ConvPE uses fixed spatial axes and produces spatial PE."""
        layer = ConvPE(channels=16)
        layer.build((None, 16, 16, 3))

        x = tf.random.normal((1, 16, 16, 3))
        output = layer(x)

        # Test actual behavior: spatial dimensions should be preserved, channels increased
        expected_shape = (1, 16, 16, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected spatial PE output shape {expected_shape}, got {output.shape}"

        # Test that spatial dimensions are preserved
        assert output.shape[1:3] == (
            16,
            16,
        ), f"Spatial dimensions should be (16, 16), got {output.shape[1:3]}"

    def test_convpe_with_activation(self):
        """Test ConvPE with activation produces correct output shape and behavior."""
        layer = ConvPE(activation=STANDARD_NONNEGATIVE_ACTIVATION, channels=16)
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((1, 8, 8, 3))
        output = layer(x)

        # Test actual behavior: correct shape with activation
        expected_shape = (1, 8, 8, 3 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test activation behavior: PE channels should be non-negative
        pe_channels = output[:, :, :, 3:]
        assert (
            tf.reduce_min(pe_channels) >= 0
        ), "Activation should produce non-negative values"

    def test_convpe_build_creates_spatial_embeddings(self):
        """Test ConvPE creates spatial embeddings with correct behavior."""
        layer = ConvPE(channels=24)
        input_shape = (None, 16, 24, 3)
        layer.build(input_shape)

        x = tf.random.normal((2, 16, 24, 3))
        output = layer(x)

        # Test actual behavior: spatial output with correct dimensions
        expected_shape = (2, 16, 24, 3 + 24)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test that embeddings are actually added
        assert (
            output.shape[-1] == 27
        ), f"Expected 27 channels total (3+24), got {output.shape[-1]}"

        # Test that input data is modified by checking PE channels are not zero
        pe_channels = output[:, :, :, 3:]  # Get the PE channels (after first 3)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "ConvPE should create non-zero PE channels"
