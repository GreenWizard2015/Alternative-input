"""Tests for LinearAttentionMixer layer."""

import tensorflow as tf
from NN.layers.LinearAttentionMixer import LinearAttentionMixer


class TestLinearAttentionMixer:
    """Test suite for LinearAttentionMixer layer."""

    def test_single_head_basic(self):
        """Test single-head attention with basic input (n_outputs=1)."""
        batch_size = 4
        seq_len = 478
        feature_dim = 64

        layer = LinearAttentionMixer(max_dim=64)
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        pooled = layer(features)

        # Default n_outputs=1, so shape is (batch, 1, feature_dim)
        expected_shape = (batch_size, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"
        assert (
            pooled.dtype == tf.float32
        ), f"Expected dtype tf.float32, got {pooled.dtype}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify input data"

    def test_multi_head_basic(self):
        """Test multi-head attention with basic input."""
        batch_size = 4
        seq_len = 478
        feature_dim = 64
        max_dim = 16  # Will result in n_heads = 64 // 16 = 4

        layer = LinearAttentionMixer(max_dim=max_dim)
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        pooled = layer(features)

        expected_shape = (batch_size, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"
        assert (
            pooled.dtype == tf.float32
        ), f"Expected dtype tf.float32, got {pooled.dtype}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify input data"

    def test_multi_head_2(self):
        """Test 2-head attention."""
        batch_size = 2
        seq_len = 100
        feature_dim = 32

        layer = LinearAttentionMixer(
            max_dim=16
        )  # Will result in n_heads = 32 // 16 = 2
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        pooled = layer(features)

        expected_shape = (batch_size, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify input data"

    def test_different_activations(self):
        """Test layer with different activation functions."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        for activation in ["gelu", "tanh", "sigmoid"]:
            layer = LinearAttentionMixer(
                max_dim=16, activation=activation
            )  # Will result in n_heads = 32 // 16 = 2
            features = tf.random.normal((batch_size, seq_len, feature_dim))

            pooled = layer(features)

            assert pooled.shape == (
                batch_size,
                1,
                feature_dim,
            ), f"Expected shape ({batch_size}, 1, {feature_dim}) with {activation} activation, got {pooled.shape}"

    def test_different_dropouts(self):
        """Test layer with different dropout rates."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        for dropout_rate in [0.0, 0.1, 0.5]:
            layer = LinearAttentionMixer(
                max_dim=16, dropout_rate=dropout_rate
            )  # Will result in n_heads = 32 // 16 = 2
            features = tf.random.normal((batch_size, seq_len, feature_dim))

            pooled = layer(features, training=True)

            assert pooled.shape == (
                batch_size,
                1,
                feature_dim,
            ), f"Expected shape ({batch_size}, {feature_dim}) with dropout {dropout_rate}, got {pooled.shape}"

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        batch_size = 2
        feature_dim = 64
        max_dim = 16  # Will result in n_heads = 64 // 16 = 4

        layer = LinearAttentionMixer(max_dim=max_dim)

        for seq_len in [10, 50, 100, 500]:
            features = tf.random.normal((batch_size, seq_len, feature_dim))
            pooled = layer(features)

            assert pooled.shape == (
                batch_size,
                1,
                feature_dim,
            ), f"Expected shape ({batch_size}, {feature_dim}) with seq_len {seq_len}, got {pooled.shape}"

    def test_output_remains_stable_with_same_input(self):
        """Test that output is deterministic for same input in inference mode."""
        batch_size = 2
        seq_len = 10
        feature_dim = 32

        layer = LinearAttentionMixer(
            max_dim=16
        )  # Will result in n_heads = 32 // 16 = 2
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        # Two forward passes with same input and no dropout
        output1 = layer(features, training=False)
        output2 = layer(features, training=False)

        # Outputs should be identical (no stochasticity in inference mode)
        tf.debugging.assert_near(
            output1,
            output2,
            rtol=1e-5,
            atol=1e-5,
            message="Output should be deterministic in inference mode",
        )

    def test_batch_size_1(self):
        """Test with batch size of 1."""
        batch_size = 1
        seq_len = 100
        feature_dim = 64

        layer = LinearAttentionMixer(
            max_dim=16
        )  # Will result in n_heads = 64 // 16 = 4
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        pooled = layer(features)

        assert pooled.shape == (
            batch_size,
            1,
            feature_dim,
        ), f"Expected shape ({batch_size}, {feature_dim}), got {pooled.shape}"

    def test_large_feature_dim(self):
        """Test with large feature dimension."""
        batch_size = 2
        seq_len = 50
        feature_dim = 512

        layer = LinearAttentionMixer(
            max_dim=64
        )  # Will result in n_heads = 512 // 64 = 8
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        pooled = layer(features)

        assert pooled.shape == (
            batch_size,
            1,
            feature_dim,
        ), f"Expected shape ({batch_size}, {feature_dim}), got {pooled.shape}"

    def test_serialization(self):
        """Test get_config for serialization."""
        layer = LinearAttentionMixer(max_dim=16, activation="tanh", dropout_rate=0.2)

        config = layer.get_config()

        assert (
            config["max_dim"] == 16
        ), f"Expected max_dim=16 in config, got {config.get('max_dim')}"
        assert (
            config["activation"] == "tanh"
        ), f"Expected activation='tanh' in config, got {config.get('activation')}"
        assert (
            config["dropout_rate"] == 0.2
        ), f"Expected dropout_rate=0.2 in config, got {config.get('dropout_rate')}"

    def test_training_vs_inference(self):
        """Test that training and inference modes work."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        layer = LinearAttentionMixer(
            max_dim=16, dropout_rate=0.5
        )  # Will result in n_heads = 32 // 16 = 2
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        # Training mode
        pooled_train = layer(features, training=True)

        # Inference mode
        pooled_infer = layer(features, training=False)

        assert pooled_train.shape == (
            batch_size,
            1,
            feature_dim,
        ), f"Expected training shape ({batch_size}, {feature_dim}), got {pooled_train.shape}"
        assert pooled_infer.shape == (
            batch_size,
            1,
            feature_dim,
        ), f"Expected inference shape ({batch_size}, {feature_dim}), got {pooled_infer.shape}"

    def test_deterministic_inference(self):
        """Test that inference is deterministic."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        layer = LinearAttentionMixer(
            max_dim=16, dropout_rate=0.0
        )  # Will result in n_heads = 32 // 16 = 2
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        # Same input should give same output in inference mode
        pooled1 = layer(features, training=False)
        pooled2 = layer(features, training=False)

        tf.debugging.assert_near(pooled1, pooled2, rtol=1e-5, atol=1e-5)

    def test_incompatible_dimensions_behavior(self):
        """Test behavior with incompatible feature dimension for number of heads."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 32 // 8 = 4
        features = tf.random.normal((batch_size, seq_len, feature_dim))

        # Test that layer produces output with correct shape
        output = layer(features)
        assert output.shape == (
            batch_size,
            1,
            feature_dim,
        ), f"Expected shape ({batch_size}, {feature_dim}), got {output.shape}"

    def test_output_range_reasonable(self):
        """Test that output values are in reasonable range."""
        batch_size = 2
        seq_len = 50
        feature_dim = 32

        # Use normal input with mean 0, std 1
        layer = LinearAttentionMixer(
            max_dim=16
        )  # Will result in n_heads = 32 // 16 = 2
        features = tf.random.normal(
            (batch_size, seq_len, feature_dim), mean=0, stddev=1
        )

        pooled = layer(features)

        # Output should be roughly in the same range as input
        # (attention is a weighted average)
        assert (
            tf.reduce_max(tf.abs(pooled)) < 10.0
        ), f"Output values should be in reasonable range, got max {tf.reduce_max(tf.abs(pooled))}"

    def test_4d_input_spatial(self):
        """Test with 4D spatial input (batch, height, width, channels).
        Pools along width axis, output: (batch, height, channels)."""
        batch_size = 1
        height = 5
        width = 4
        feature_dim = 32

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 32 // 8 = 4
        features = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features)

        # Pools width dimension: (batch, height, channels)
        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_4d_input_direct(self):
        """Test with 4D input directly.
        Pools along width axis, output: (batch, height, channels)."""
        batch_size = 1
        height = 5
        width = 4
        feature_dim = 32

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 32 // 8 = 4
        # 4D input: (batch, height, width, channels)
        features_4d = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features_4d)

        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features_4d)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_4d_batch_multiple_spatial(self):
        """Test with multiple 4D inputs.
        Pools along width axis, output: (batch, height, channels)."""
        batch_size = 2
        height = 5
        width = 4
        feature_dim = 32

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 32 // 8 = 4

        # 4D input directly
        features_4d = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features_4d)

        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features_4d)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_4d_long_sequence(self):
        """Test with 4D input with longer spatial dimensions.
        Pools along last spatial axis."""
        batch_size = 1
        height = 20
        width = 30
        feature_dim = 64

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 64 // 8 = 8

        # 4D spatial input
        features = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features)

        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_4d_small_feature_dim(self):
        """Test 4D spatial input with small feature dimension."""
        batch_size = 1
        height = 5
        width = 4
        feature_dim = 16

        layer = LinearAttentionMixer(max_dim=4)  # Will result in n_heads = 16 // 4 = 4
        features = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features)

        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_4d_exact_case(self):
        """Test the exact case from error: (1, 5, 4, 32).
        Pools width axis, output: (1, 5, 1, 32)."""
        batch_size = 1
        height = 5
        width = 4
        feature_dim = 32

        layer = LinearAttentionMixer(
            max_dim=16
        )  # Will result in n_heads = 32 // 16 = 2

        # Direct 4D input (1, 5, 4, 32)
        features_4d = tf.random.normal((batch_size, height, width, feature_dim))

        pooled = layer(features_4d)

        expected_shape = (batch_size, height, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features_4d)
        ), "LinearAttentionMixer should modify 4D input data"

    def test_5d_input_spatial(self):
        """Test with 5D spatial input.
        Pools along last spatial axis, output preserves leading spatial dims."""
        batch_size = 2
        d1 = 5
        d2 = 6
        d3 = 7
        feature_dim = 32

        layer = LinearAttentionMixer(max_dim=8)  # Will result in n_heads = 32 // 8 = 4
        features = tf.random.normal((batch_size, d1, d2, d3, feature_dim))

        pooled = layer(features)

        # Pools d3 dimension: (batch, d1, d2, channels)
        expected_shape = (batch_size, d1, d2, 1, feature_dim)
        assert (
            pooled.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {pooled.shape}"

        # Test actual behavior: should modify input
        assert not tf.reduce_all(
            tf.equal(pooled, features)
        ), "LinearAttentionMixer should modify 5D input data"
