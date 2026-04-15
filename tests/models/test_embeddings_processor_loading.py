"""Tests for EmbeddingsProcessor eager loading behavior."""

import tensorflow as tf
from NN.models.EmbeddingsProcessor import EmbeddingsProcessor


class TestEmbeddingsProcessorLoading:
    """Test EmbeddingsProcessor eager loading functionality."""

    def test_initialization_dense_produces_valid_output(self):
        """Test that EmbeddingsProcessor with dense method produces valid output."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="dense")

        # Create test inputs
        concatenated = tf.random.normal([2, 1, 5 * 64])
        shape = tf.constant([2, 10])

        # Test that it produces valid output
        result = processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )

        assert isinstance(
            result, tf.Tensor
        ), "Dense processor should produce tensor output"
        assert result.shape == [
            2,
            10,
            64,
        ], f"Dense processor expected shape [2, 10, 64], got {result.shape}"
        assert result.dtype == tf.float32, "Dense processor output should be float32"

    def test_initialization_attention_produces_valid_output(self):
        """Test that EmbeddingsProcessor with attention method produces valid output."""
        processor = EmbeddingsProcessor(embedding_size=128, mixing_method="attention")

        # Create test inputs
        concatenated = tf.random.normal([2, 1, 5 * 128])
        shape = tf.constant([2, 10])

        # Test that it produces valid output
        result = processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )

        assert isinstance(
            result, tf.Tensor
        ), "Attention processor should produce tensor output"
        assert result.shape == [
            2,
            10,
            128,
        ], f"Attention processor expected shape [2, 10, 128], got {result.shape}"
        assert (
            result.dtype == tf.float32
        ), "Attention processor output should be float32"

    def test_invalid_mixing_method_fails_initialization(self):
        """Test that invalid mixing method produces no valid output."""
        processor = None
        try:
            processor = EmbeddingsProcessor(
                embedding_size=64, mixing_method="invalid_method"
            )
        except (ValueError, KeyError, TypeError):
            # Expected behavior: initialization fails with invalid method
            pass

        if processor is not None:
            # If it somehow was created, verify it doesn't work as expected
            concatenated = tf.random.normal([2, 1, 5 * 64])
            shape = tf.constant([2, 10])
            call_succeeded = False
            try:
                processor.call(concatenated_embeddings=concatenated, shape=shape)
                call_succeeded = True
            except (ValueError, KeyError, AttributeError):
                pass

            assert (
                not call_succeeded
            ), "EmbeddingsProcessor should not accept invalid mixing method"

    def test_default_parameters_produces_output(self):
        """Test that EmbeddingsProcessor with defaults produces valid output."""
        processor = EmbeddingsProcessor()

        # Test with default parameters - should work with attention method and embedding_size=64
        concatenated = tf.random.normal([2, 1, 5 * 64])
        shape = tf.constant([2, 10])
        result = processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )

        assert isinstance(
            result, tf.Tensor
        ), "Should produce tensor output with defaults"
        assert result.shape == [
            2,
            10,
            64,
        ], f"Expected shape [2, 10, 64], got {result.shape}"

    def test_custom_name_initialization(self):
        """Test EmbeddingsProcessor initialization with custom name."""
        custom_name = "test_processor"
        processor = EmbeddingsProcessor(name=custom_name)

        assert (
            processor.name == custom_name
        ), f"Custom name should be '{custom_name}', got '{processor.name}'"

    def test_tensorflow_layer_inheritance(self):
        """Test that EmbeddingsProcessor properly inherits from tf.keras.layers.Layer."""
        processor = EmbeddingsProcessor()

        assert isinstance(
            processor, tf.keras.layers.Layer
        ), "EmbeddingsProcessor should inherit from tf.keras.layers.Layer"
        assert hasattr(processor, "call"), "EmbeddingsProcessor should have call method"
        assert hasattr(
            processor, "get_config"
        ), "EmbeddingsProcessor should have get_config method"

    def test_different_mixing_methods_produce_output(self):
        """Test that different mixing methods produce valid output."""
        # Test dense method
        dense_processor = EmbeddingsProcessor(mixing_method="dense")
        concatenated = tf.random.normal([2, 1, 5 * 64])
        shape = tf.constant([2, 10])

        result_dense = dense_processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )
        assert result_dense.shape == [
            2,
            10,
            64,
        ], f"Dense method should produce shape [2, 10, 64], got {result_dense.shape}"

        # Test attention method
        attention_processor = EmbeddingsProcessor(mixing_method="attention")
        result_attention = attention_processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )
        assert result_attention.shape == [
            2,
            10,
            64,
        ], f"Attention method should produce shape [2, 10, 64], got {result_attention.shape}"

    def test_final_dense_layer_applies_activation(self):
        """Test that final dense layer applies activation correctly."""
        processor = EmbeddingsProcessor(embedding_size=128)

        # Test that the final dense layer produces output in expected range
        # (tanh activation should produce values in [-1, 1])
        concatenated = tf.random.normal([2, 1, 5 * 128])
        shape = tf.constant([2, 10])
        result = processor.call(
            concatenated_embeddings=concatenated,
            shape=shape,
        )

        # Check that output is within tanh range (approximately [-1, 1])
        # Focus on behavior: the activation should constrain values to expected range
        assert tf.reduce_all(result >= -1.1), "Output should be >= -1.1"
        assert tf.reduce_all(result <= 1.1), "Output should be <= 1.1"

    def test_dense_mode_handles_variable_batch_sizes(self):
        """Test that dense mode works with different batch sizes."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="dense")

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            concatenated = tf.random.normal([batch_size, 1, 5 * 64])
            shape = tf.constant([batch_size, 10])

            result = processor.call(
                concatenated_embeddings=concatenated,
                shape=shape,
            )

            assert result.shape == [
                batch_size,
                10,
                64,
            ], f"Dense mode should handle batch_size={batch_size}, but got shape {result.shape}"

    def test_dense_mode_produces_correct_output_shape(self):
        """Test that dense mode produces output with correct shape for various inputs."""
        processor = EmbeddingsProcessor(embedding_size=32, mixing_method="dense")

        # Test with various embedding sizes and batch configurations
        test_cases = [
            (1, 5),  # batch=1, timesteps=5
            (4, 20),  # batch=4, timesteps=20
            (8, 15),  # batch=8, timesteps=15
        ]

        for batch_size, timesteps in test_cases:
            concatenated = tf.random.normal([batch_size, 1, 5 * 32])
            shape = tf.constant([batch_size, timesteps])

            result = processor.call(
                concatenated_embeddings=concatenated,
                shape=shape,
            )

            assert result.shape == [
                batch_size,
                timesteps,
                32,
            ], f"Dense mode should produce shape [{batch_size}, {timesteps}, 32]"

    def test_attention_mode_produces_correct_output_shape(self):
        """Test that attention mode produces output with correct shape for various inputs."""
        processor = EmbeddingsProcessor(embedding_size=32, mixing_method="attention")

        # Test with various batch sizes and timesteps
        test_cases = [
            (1, 5),  # batch=1, timesteps=5
            (4, 20),  # batch=4, timesteps=20
            (8, 15),  # batch=8, timesteps=15
        ]

        for batch_size, timesteps in test_cases:
            concatenated = tf.random.normal([batch_size, 1, 5 * 32])
            shape = tf.constant([batch_size, timesteps])

            result = processor.call(
                concatenated_embeddings=concatenated,
                shape=shape,
            )

            assert result.shape == [
                batch_size,
                timesteps,
                32,
            ], f"Attention mode should produce shape [{batch_size}, {timesteps}, 32]"

    def test_immediate_output_without_prior_calls(self):
        """Test that processor produces valid output immediately after initialization."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Create individual embeddings and concatenate them
        individual_embeddings = [
            tf.random.normal([2, 1, 64]),  # userId
            tf.random.normal([2, 1, 64]),  # screenId
            tf.random.normal([2, 1, 64]),  # cameraId
            tf.random.normal([2, 1, 64]),  # monitorId
            tf.random.normal([2, 1, 64]),  # placeId
        ]
        concatenated_embeddings = tf.concat(individual_embeddings, axis=-1)

        # First call should work without any prior calls
        result = processor.call(
            concatenated_embeddings=concatenated_embeddings,
            shape=tf.constant([2, 10]),
        )

        # Should produce valid output immediately
        assert isinstance(result, tf.Tensor), "Should produce tensor output"
        assert result.shape == [
            2,
            10,
            64,
        ], f"Expected shape [2, 10, 64], got {result.shape}"
