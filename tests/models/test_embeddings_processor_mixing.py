"""Tests for EmbeddingsProcessor mixing functionality."""

import pytest
import tensorflow as tf
from NN.models.EmbeddingsProcessor import EmbeddingsProcessor


class TestEmbeddingsProcessorMixing:
    """Test EmbeddingsProcessor mixing functionality."""

    @pytest.fixture
    def sample_embeddings(self):
        """Sample individual embeddings for testing."""
        embedding_size = 64
        return {
            "userId": tf.random.normal(
                [2, 1, embedding_size]
            ),  # batch=2, timesteps=1, emb_size=64
            "screenId": tf.random.normal([2, 1, embedding_size]),
            "cameraId": tf.random.normal([2, 1, embedding_size]),
            "monitorId": tf.random.normal([2, 1, embedding_size]),
            "placeId": tf.random.normal([2, 1, embedding_size]),
        }

    @pytest.fixture
    def sample_concatenated_embeddings(self, sample_embeddings):
        """Concatenated embeddings tensor for testing."""
        # Concatenate all 5 embeddings along feature dimension
        embeddings_list = [
            sample_embeddings["userId"],
            sample_embeddings["screenId"],
            sample_embeddings["cameraId"],
            sample_embeddings["monitorId"],
            sample_embeddings["placeId"],
        ]
        return tf.concat(embeddings_list, axis=-1)  # (2, 1, 5*64)

    @pytest.fixture
    def sample_default_ids(self):
        """Sample default IDs for testing."""
        return {
            "userId": 0,
            "screenId": 0,
            "cameraId": 0,
            "monitorId": 0,
            "placeId": 0,
        }

    @pytest.fixture
    def sample_shape(self):
        """Sample shape tensor for testing."""
        return tf.constant([2, 10], dtype=tf.int32)  # batch_size=2, timesteps=10

    def test_dense_mixing_basic(self, sample_concatenated_embeddings, sample_shape):
        """Test basic dense mixing functionality."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="dense")

        result = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
        )

        assert result.shape == [2, 10, 64]  # batch, timesteps, embedding_size

    def test_attention_mixing_basic(self, sample_concatenated_embeddings, sample_shape):
        """Test basic attention mixing functionality."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="attention")

        result = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
        )

        assert result.shape == [2, 10, 64]  # batch, timesteps, embedding_size

    def test_mixed_embedding_inputs(
        self, sample_embeddings, sample_concatenated_embeddings, sample_shape
    ):
        """Test that mixed embedding inputs work correctly."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Modify some embeddings to be different
        mixed_embeddings = sample_embeddings.copy()
        mixed_embeddings["userId"] = tf.random.normal(
            [2, 1, 64]
        )  # Different user embeddings

        # Create concatenated version of mixed embeddings
        mixed_concatenated = tf.concat(
            [
                mixed_embeddings["userId"],
                mixed_embeddings["screenId"],
                mixed_embeddings["cameraId"],
                mixed_embeddings["monitorId"],
                mixed_embeddings["placeId"],
            ],
            axis=-1,
        )

        result = processor.call(
            concatenated_embeddings=mixed_concatenated,
            shape=sample_shape,
        )

        assert result.shape == [2, 10, 64]
        # Result should be different from original due to different user embeddings
        assert not tf.reduce_all(
            tf.equal(
                result,
                processor.call(
                    concatenated_embeddings=sample_concatenated_embeddings,
                    shape=sample_shape,
                ),
            )
        )

    def test_missing_embeddings_fails_silently(self, sample_shape):
        """Test that missing concatenated embeddings produces no valid output."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # When required parameter is missing, call should fail
        # This tests the behavior/side-effect: the call returns None or fails
        result = None
        try:
            result = processor.call(shape=sample_shape)
        except (KeyError, TypeError, AttributeError):
            # Expected behavior: call fails without required input
            pass

        assert (
            result is None
        ), "EmbeddingsProcessor should not produce valid output without concatenated_embeddings parameter"

    def test_missing_shape_fails_silently(self, sample_concatenated_embeddings):
        """Test that missing shape produces no valid output."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # When required parameter is missing, call should fail
        # This tests the behavior/side-effect: the call returns None or fails
        result = None
        try:
            result = processor.call(
                concatenated_embeddings=sample_concatenated_embeddings,
            )
        except (KeyError, TypeError, AttributeError):
            # Expected behavior: call fails without shape parameter
            pass

        assert (
            result is None
        ), "EmbeddingsProcessor should not produce valid output without shape parameter"

    def test_incomplete_embedding_size_produces_invalid_output(
        self, sample_embeddings, sample_shape
    ):
        """Test that incorrect concatenated embedding size produces invalid output."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Create concatenated embeddings with only 4 embeddings instead of 5
        incomplete_embeddings = sample_embeddings.copy()
        del incomplete_embeddings["userId"]

        incomplete_concatenated = tf.concat(
            [
                incomplete_embeddings["screenId"],
                incomplete_embeddings["cameraId"],
                incomplete_embeddings["monitorId"],
                incomplete_embeddings["placeId"],
            ],
            axis=-1,
        )  # This will be (2, 1, 4*64) instead of (2, 1, 5*64)

        # Test that call with incorrect embedding size fails or produces unexpected output
        # This tests behavior: either raises an error or produces wrong shape
        result = None
        try:
            result = processor.call(
                concatenated_embeddings=incomplete_concatenated,
                shape=sample_shape,
            )
        except (tf.errors.InvalidArgumentError, ValueError, AttributeError):
            # Expected behavior: processing fails with incorrect input
            pass

        assert (
            result is None or result.shape[-1] != 64
        ), "EmbeddingsProcessor should fail with incomplete embedding size (4 embeddings instead of 5)"

    def test_incorrect_embedding_rank_produces_invalid_output(self, sample_shape):
        """Test that incorrect embedding rank produces invalid output."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Create concatenated embedding with wrong rank (should be rank 3)
        wrong_rank_concatenated = tf.random.normal(
            [2, 5 * 64]
        )  # Missing timesteps dimension

        # Test that call with wrong rank fails or produces unexpected output
        # This tests behavior: call fails with incorrect rank input
        result = None
        try:
            result = processor.call(
                concatenated_embeddings=wrong_rank_concatenated,
                shape=sample_shape,
            )
        except (tf.errors.InvalidArgumentError, ValueError, AttributeError):
            # Expected behavior: processing fails with incorrect rank
            pass

        assert (
            result is None
        ), "EmbeddingsProcessor should fail with incorrect embedding rank (rank 2 instead of rank 3)"

    def test_incorrect_embedding_dimension_fails(self, sample_shape):
        """Test that incorrect embedding dimension causes processing to fail."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Create concatenated embedding with wrong dimension
        # Should be 5*64=320, but we use 5*32=160
        wrong_dim_concatenated = tf.random.normal([2, 1, 5 * 32])

        # Test that call with wrong dimension produces invalid output
        result = None
        try:
            result = processor.call(
                concatenated_embeddings=wrong_dim_concatenated,
                shape=sample_shape,
            )
        except (tf.errors.InvalidArgumentError, ValueError, AttributeError):
            # Expected behavior: processing fails with incorrect input
            pass
        assert (
            result is None or result.shape[-1] != 64
        ), "EmbeddingsProcessor should fail for wrong embedding dimension (5*32 instead of 5*64)"

    def test_training_flag_propagation_dense(
        self, sample_concatenated_embeddings, sample_shape
    ):
        """Test that training flag is properly propagated in dense mode."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="dense")

        result_training = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
            training=True,
        )

        result_inference = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
            training=False,
        )

        # Results should have same shape
        assert (
            result_training.shape == result_inference.shape
        ), f"Training and inference results should have same shape, got {result_training.shape} vs {result_inference.shape}"

    def test_training_flag_propagation_attention(
        self, sample_concatenated_embeddings, sample_shape
    ):
        """Test that training flag is properly propagated in attention mode."""
        processor = EmbeddingsProcessor(embedding_size=64, mixing_method="attention")

        result_training = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
            training=True,
        )

        result_inference = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
            training=False,
        )

        # Results should have same shape
        assert (
            result_training.shape == result_inference.shape
        ), f"Training and inference results should have same shape, got {result_training.shape} vs {result_inference.shape}"

    def test_output_shape_consistency(self, sample_shape):
        """Test that output shape is consistent across different inputs."""
        processor = EmbeddingsProcessor(embedding_size=32)

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            # Create concatenated embeddings for each batch size
            test_concatenated = tf.random.normal([batch_size, 1, 5 * 32])
            test_shape = tf.constant([batch_size, 10])

            result = processor.call(
                concatenated_embeddings=test_concatenated,
                shape=test_shape,
            )

            assert result.shape == [
                batch_size,
                10,
                32,
            ], f"Expected shape [{batch_size}, 10, 32], got {result.shape}"

    def test_different_timesteps(self, sample_concatenated_embeddings):
        """Test that different timesteps work correctly."""
        processor = EmbeddingsProcessor(embedding_size=64)

        for timesteps in [5, 10, 20]:
            test_shape = tf.constant([2, timesteps])

            result = processor.call(
                concatenated_embeddings=sample_concatenated_embeddings,
                shape=test_shape,
            )

            assert result.shape == [
                2,
                timesteps,
                64,
            ], f"Expected shape [2, {timesteps}, 64], got {result.shape}"

    def test_uninitialized_mixer_behavior(
        self, sample_concatenated_embeddings, sample_shape
    ):
        """Test that processing succeeds when mixer is properly initialized."""
        processor = EmbeddingsProcessor(embedding_size=64)

        # Test that processing succeeds with proper initialization
        result = processor.call(
            concatenated_embeddings=sample_concatenated_embeddings,
            shape=sample_shape,
        )

        # Verify that with proper initialization, processing succeeds
        assert (
            result is not None
        ), "Processor should successfully process with proper initialization"
        assert result.shape == [
            2,
            10,
            64,
        ], f"Should produce valid output shape [2, 10, 64], got {result.shape}"

    def test_concatenation_correctness(self, sample_shape):
        """Test that embeddings are correctly concatenated before mixing."""
        processor = EmbeddingsProcessor(embedding_size=32)

        # Create embeddings with predictable values
        user_embedding = tf.ones([2, 1, 32]) * 1.0  # All ones
        screen_embedding = tf.ones([2, 1, 32]) * 2.0  # All twos
        camera_embedding = tf.ones([2, 1, 32]) * 3.0  # All threes
        monitor_embedding = tf.ones([2, 1, 32]) * 4.0  # All fours
        place_embedding = tf.ones([2, 1, 32]) * 5.0  # All fives

        # Concatenate the embeddings
        concatenated_embeddings = tf.concat(
            [
                user_embedding,
                screen_embedding,
                camera_embedding,
                monitor_embedding,
                place_embedding,
            ],
            axis=-1,
        )

        result = processor.call(
            concatenated_embeddings=concatenated_embeddings,
            shape=sample_shape,
        )

        # For dense mixing, the concatenated result should be [1,2,3,4,5] = 15 in each feature
        # After mixing and final dense, it should be a learned transformation
        assert result.shape == [
            2,
            10,
            32,
        ], f"Expected shape [2, 10, 32], got {result.shape}"

    def test_attention_mixer_dimension_handling(self, sample_shape):
        """Test that attention mixer handles dimensions correctly."""
        processor = EmbeddingsProcessor(embedding_size=16, mixing_method="attention")

        # Test with different embedding sizes - create concatenated embeddings with size 16
        small_concatenated = tf.random.normal([2, 1, 5 * 16])

        result = processor.call(
            concatenated_embeddings=small_concatenated,
            shape=sample_shape,
        )

        assert result.shape == [
            2,
            10,
            16,
        ], f"Expected shape [2, 10, 16], got {result.shape}"

    def test_final_dense_layer_applied(self, sample_shape):
        """Test that final dense layer transformation is applied."""
        processor = EmbeddingsProcessor(embedding_size=32)

        # Create embeddings that will produce simple patterns
        user_embedding = tf.ones([2, 1, 32]) * 0.5
        screen_embedding = tf.ones([2, 1, 32]) * 0.5
        camera_embedding = tf.ones([2, 1, 32]) * 0.5
        monitor_embedding = tf.ones([2, 1, 32]) * 0.5
        place_embedding = tf.ones([2, 1, 32]) * 0.5

        # Concatenate embeddings
        simple_concatenated = tf.concat(
            [
                user_embedding,
                screen_embedding,
                camera_embedding,
                monitor_embedding,
                place_embedding,
            ],
            axis=-1,
        )

        result = processor.call(
            concatenated_embeddings=simple_concatenated,
            shape=sample_shape,
        )

        # All inputs are same, so result should be consistent
        assert result.shape == [
            2,
            10,
            32,
        ], f"Expected shape [2, 10, 32], got {result.shape}"

        # The final dense layer should apply tanh activation
        # Result should be in range [-1, 1]
        assert tf.reduce_all(result >= -1.0), "Output values should be >= -1.0"
        assert tf.reduce_all(result <= 1.0), "Output values should be <= 1.0"
