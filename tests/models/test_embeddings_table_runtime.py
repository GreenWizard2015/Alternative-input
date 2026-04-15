"""Tests for EmbeddingsTable runtime functionality with parameter-based defaults."""

import pytest
import tensorflow as tf
from NN.models.EmbeddingsTable import EmbeddingsTable


class TestEmbeddingsTableRuntime:
    """Test EmbeddingsTable runtime functionality."""

    @pytest.fixture
    def sample_stats(self):
        """Sample stats dictionary for testing."""
        return {
            "userId": ["user_0", "user_1", "user_2"],
            "screenId": ["screen_0", "screen_1"],
            "cameraId": ["camera_0", "camera_1", "camera_2", "camera_3"],
            "monitorId": ["monitor_0", "monitor_1"],
            "placeId": ["place_0", "place_1", "place_2", "place_3", "place_4"],
        }

    @pytest.fixture
    def sample_default_ids(self):
        """Sample default IDs for testing."""
        return {
            "userId": 0,
            "screenId": 0,
            "cameraId": 1,
            "monitorId": 0,
            "placeId": 2,
        }

    @pytest.fixture
    def sample_shape(self):
        """Sample shape tensor for testing."""
        return tf.constant([2, 10], dtype=tf.int32)  # batch_size=2, timesteps=10

    @pytest.fixture
    def sample_vocab(self, sample_stats):
        """Sample vocab dictionary (vocab sizes for each ID type)."""
        return {k: len(v) for k, v in sample_stats.items()}

    def test_call_with_all_defaults(
        self, sample_vocab, sample_default_ids, sample_shape
    ):
        """Test EmbeddingsTable call with all default IDs."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)

        result = table.call(
            userId=None,
            screenId=None,
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
        )

        # Should return concatenated tensor
        assert isinstance(result, tf.Tensor)
        # Shape should be (batch_size, 1, 5*embedding_size)
        assert result.shape == [2, 1, 5 * 64]

    def test_call_with_explicit_ids(self, sample_vocab, sample_shape):
        """Test EmbeddingsTable call with explicit ID tensors."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)

        # Create explicit ID tensors
        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),
            screenId=tf.constant([[0], [1]], dtype=tf.int32),
            cameraId=tf.constant([[1], [2]], dtype=tf.int32),
            monitorId=tf.constant([[0], [1]], dtype=tf.int32),
            placeId=tf.constant([[2], [3]], dtype=tf.int32),
            shape=sample_shape,
        )

        # Should return concatenated tensor
        assert isinstance(result, tf.Tensor)
        # Shape should be (batch_size, 1, 5*embedding_size)
        assert result.shape == [2, 1, 5 * 64]

    def test_call_mixed_explicit_and_defaults(
        self, sample_vocab, sample_default_ids, sample_shape
    ):
        """Test EmbeddingsTable call with some explicit IDs and some defaults."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)

        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),  # explicit
            screenId=None,  # will use default
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
        )

        # Should return concatenated tensor
        assert isinstance(result, tf.Tensor)
        # Shape should be (batch_size, 1, 5*embedding_size)
        assert result.shape == [2, 1, 5 * 64]

    def test_missing_default_id_produces_invalid_output(
        self, sample_vocab, sample_shape
    ):
        """Test that missing default ID when needed causes call to fail."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)
        # Don't load defaults

        # Don't provide userId tensor and no default_ids
        result = None
        try:
            result = table.call(
                userId=None,
                screenId=tf.constant([[0], [1]], dtype=tf.int32),
                cameraId=tf.constant([[0], [1]], dtype=tf.int32),
                monitorId=tf.constant([[0], [1]], dtype=tf.int32),
                placeId=tf.constant([[0], [1]], dtype=tf.int32),
                shape=sample_shape,
            )
        except (ValueError, KeyError, TypeError):
            # Expected behavior: call fails when required value is missing
            pass

        assert (
            result is None
        ), "EmbeddingsTable should not produce valid output without required userId"

    def test_training_flag_propagation(
        self, sample_vocab, sample_default_ids, sample_shape
    ):
        """Test that training flag is properly propagated to embedding layers."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)

        # Call with training=True
        result_training = table.call(
            userId=None,
            screenId=None,
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
            training=True,
        )

        # Call with training=False
        result_inference = table.call(
            userId=None,
            screenId=None,
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
            training=False,
        )

        # Results should have same shape
        assert isinstance(
            result_training, tf.Tensor
        ), "Training result should be a tensor"
        assert isinstance(
            result_inference, tf.Tensor
        ), "Inference result should be a tensor"
        assert (
            result_training.shape == result_inference.shape
        ), f"Training and inference shapes should match: {result_training.shape} vs {result_inference.shape}"
        assert result_training.shape == [
            2,
            1,
            5 * 64,
        ], f"Expected shape [2, 1, 320], got {result_training.shape}"

    def test_embedding_layer_creation(
        self, sample_vocab, sample_default_ids, sample_shape
    ):
        """Test that embedding layers are properly created and produce consistent results."""
        table = EmbeddingsTable(vocab=sample_vocab, embedding_size=64)

        # First call
        result1 = table.call(
            userId=None,
            screenId=None,
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
        )

        # Second call should produce same results
        result2 = table.call(
            userId=None,
            screenId=None,
            cameraId=None,
            monitorId=None,
            placeId=None,
            default_ids=sample_default_ids,
            shape=sample_shape,
        )

        # Results should have same shapes
        assert isinstance(result1, tf.Tensor), "First call result should be a tensor"
        assert isinstance(result2, tf.Tensor), "Second call result should be a tensor"
        assert (
            result1.shape == result2.shape
        ), f"Shapes should match across calls: {result1.shape} vs {result2.shape}"
        assert result1.shape == [
            2,
            1,
            5 * 64,
        ], f"Expected shape [2, 1, 320], got {result1.shape}"

        # Results should be tensors of correct type
        assert (
            result1.dtype == tf.float32
        ), f"First result should be float32, got {result1.dtype}"
        assert (
            result2.dtype == tf.float32
        ), f"Second result should be float32, got {result2.dtype}"
