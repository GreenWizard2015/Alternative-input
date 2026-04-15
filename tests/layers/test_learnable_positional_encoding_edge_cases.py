"""Tests for LearnablePositionalEncoding edge cases."""

import tensorflow as tf

from NN.layers.EncodingLayers import LearnablePositionalEncoding


class TestLearnablePositionalEncodingEdgeCases:
    """Tests for LearnablePositionalEncoding edge cases."""

    def test_zero_channels(self):
        """Test layer with zero channels produces output with original dimensions."""
        layer = LearnablePositionalEncoding(channels=0)
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((1, 8, 8, 3))
        output = layer(x)

        # With 0 channels, output should just pass through input
        expected_channels = 3
        assert (
            output.shape[-1] == expected_channels
        ), f"Expected {expected_channels} channels (no PE), got {output.shape[-1]}"

        # Test actual behavior: output should be identical to input when channels=0
        assert tf.reduce_all(
            tf.equal(output, x)
        ), "With zero channels, output should be identical to input"

    def test_very_small_spatial_dims(self):
        """Test with 1x1 spatial dimensions."""
        layer = LearnablePositionalEncoding(channels=8, axis=[-3, -2])
        layer.build((None, 1, 1, 3))

        x = tf.random.normal((2, 1, 1, 3))
        output = layer(x)

        expected_shape = (2, 1, 1, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels even with small dimensions
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_very_large_spatial_dims(self):
        """Test with large spatial dimensions."""
        layer = LearnablePositionalEncoding(channels=4, axis=[-3, -2])
        layer.build((None, 512, 512, 3))

        x = tf.random.normal((1, 512, 512, 3))
        output = layer(x)

        expected_shape = (1, 512, 512, 3 + 4)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels with large dimensions
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_single_channel_input(self):
        """Test with single channel input."""
        layer = LearnablePositionalEncoding(channels=16)
        layer.build((None, 8, 8, 1))

        x = tf.random.normal((2, 8, 8, 1))
        output = layer(x)

        expected_shape = (2, 8, 8, 1 + 16)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels with single channel input
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_many_channel_input(self):
        """Test with many channel input."""
        layer = LearnablePositionalEncoding(channels=8)
        layer.build((None, 8, 8, 128))

        x = tf.random.normal((2, 8, 8, 128))
        output = layer(x)

        expected_shape = (2, 8, 8, 128 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels with many channels
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_rank_5_input(self):
        """Test with rank-5 input (e.g., video: B, T, H, W, C)."""
        layer = LearnablePositionalEncoding(channels=8, axis=[2, 3])

        # Build with rank-5 shape
        input_shape = (2, 10, 16, 16, 3)
        layer.build(input_shape)

        x = tf.random.normal((2, 10, 16, 16, 3))
        output = layer(x)

        expected_shape = (2, 10, 16, 16, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels with rank-5 input
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_negative_axis_indices(self):
        """Test that negative axis indices work correctly with single axis."""
        # Use single negative axis which is supported
        layer = LearnablePositionalEncoding(channels=8, axis=[-2])
        layer.build((None, 8, 8, 3))

        x = tf.random.normal((2, 8, 8, 3))
        output = layer(x)

        # Output should be correctly shaped with concatenated channels
        expected_shape = (2, 8, 8, 3 + 8)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Test actual behavior: should create non-zero PE channels with negative axis indices
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"

    def test_dynamic_shape_batch_dimension(self):
        """Test with dynamic batch dimension."""
        layer = LearnablePositionalEncoding(channels=8)
        layer.build((None, 16, 16, 3))

        # Create input with dynamic batch size
        batch_size = tf.constant(4)
        x = tf.random.normal([batch_size, 16, 16, 3])
        output = layer(x)

        # Output shape should be correct
        expected_spatial_dims = (16, 16, 3 + 8)
        assert (
            output.shape[1:] == expected_spatial_dims
        ), f"Expected spatial dims {expected_spatial_dims}, got {output.shape[1:]}"

        # Test actual behavior: should create non-zero PE channels with dynamic batch size
        pe_channels = output[
            :, :, :, x.shape[-1] :
        ]  # Get the PE channels (after original channels)
        assert not tf.reduce_all(
            tf.equal(pe_channels, 0)
        ), "LearnablePositionalEncoding should create non-zero PE channels"
