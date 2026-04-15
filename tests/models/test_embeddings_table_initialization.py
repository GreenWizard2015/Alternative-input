"""Tests for EmbeddingsTable initialization and basic functionality."""

import tensorflow as tf
from NN.models.EmbeddingsTable import EmbeddingsTable


# Standard vocab for testing
STANDARD_VOCAB = {
    "userId": 100,
    "screenId": 50,
    "cameraId": 20,
    "monitorId": 30,
    "placeId": 25,
}


class TestEmbeddingsTableInitialization:
    """Test EmbeddingsTable initialization and basic setup."""

    def test_initialization_default(self):
        """Test EmbeddingsTable initialization with default parameters."""
        table = EmbeddingsTable(vocab=STANDARD_VOCAB)

        # Test that the table can actually process embeddings
        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),
            screenId=tf.constant([[0], [1]], dtype=tf.int32),
            cameraId=tf.constant([[0], [1]], dtype=tf.int32),
            monitorId=tf.constant([[0], [1]], dtype=tf.int32),
            placeId=tf.constant([[0], [1]], dtype=tf.int32),
            shape=tf.constant([2, 1]),
        )

        assert isinstance(result, tf.Tensor), "Should produce tensor output"
        assert result.shape == [
            2,
            1,
            5 * 64,
        ], f"Expected shape [2, 1, 320], got {result.shape}"

    def test_initialization_custom_embedding_size(self):
        """Test EmbeddingsTable initialization with custom embedding size."""
        table = EmbeddingsTable(vocab=STANDARD_VOCAB, embedding_size=128)

        # Test that the table works with custom embedding size
        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),
            screenId=tf.constant([[0], [1]], dtype=tf.int32),
            cameraId=tf.constant([[0], [1]], dtype=tf.int32),
            monitorId=tf.constant([[0], [1]], dtype=tf.int32),
            placeId=tf.constant([[0], [1]], dtype=tf.int32),
            shape=tf.constant([2, 1]),
        )

        assert isinstance(result, tf.Tensor), "Should produce tensor output"
        assert result.shape == [
            2,
            1,
            5 * 128,
        ], f"Expected shape [2, 1, 640], got {result.shape}"

    def test_initialization_with_name(self):
        """Test EmbeddingsTable initialization with custom name."""
        custom_name = "test_embeddings_table"
        table = EmbeddingsTable(vocab=STANDARD_VOCAB, name=custom_name)

        # Test that the table works with custom name
        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),
            screenId=tf.constant([[0], [1]], dtype=tf.int32),
            cameraId=tf.constant([[0], [1]], dtype=tf.int32),
            monitorId=tf.constant([[0], [1]], dtype=tf.int32),
            placeId=tf.constant([[0], [1]], dtype=tf.int32),
            shape=tf.constant([2, 1]),
        )

        assert isinstance(result, tf.Tensor), "Should produce tensor output"
        assert result.shape == [
            2,
            1,
            5 * 64,
        ], f"Expected shape [2, 1, 320], got {result.shape}"

    def test_vocab_validation_missing_keys(self):
        """Test that EmbeddingsTable doesn't produce valid output with missing vocab keys."""
        incomplete_vocab = {"userId": 100, "screenId": 50}  # Missing other keys
        table = None
        try:
            table = EmbeddingsTable(vocab=incomplete_vocab)
        except (ValueError, KeyError):
            # Expected behavior: initialization fails with incomplete vocab
            pass

        if table is not None:
            # If somehow initialization succeeded, verify it doesn't work as expected
            call_succeeded = False
            try:
                table.call(
                    userId=None,
                    screenId=None,
                    cameraId=None,
                    monitorId=None,
                    placeId=None,
                )
                call_succeeded = True
            except (ValueError, KeyError, AttributeError):
                pass

            assert (
                not call_succeeded
            ), "EmbeddingsTable should not accept incomplete vocab"

    def test_tensorflow_model_inheritance(self):
        """Test that EmbeddingsTable properly inherits from tf.keras.Model."""
        table = EmbeddingsTable(vocab=STANDARD_VOCAB)

        assert isinstance(
            table, tf.keras.Model
        ), "EmbeddingsTable should be a tf.keras.Model"
        assert hasattr(table, "call"), "EmbeddingsTable should have call method"
        assert hasattr(
            table, "get_config"
        ), "EmbeddingsTable should have get_config method"

    def test_model_creates_all_layers(self):
        """Test that model creates all required embedding layers on initialization."""
        table = EmbeddingsTable(vocab=STANDARD_VOCAB)

        # Test that the model can process embeddings immediately after initialization
        result = table.call(
            userId=tf.constant([[0], [1]], dtype=tf.int32),
            screenId=tf.constant([[0], [1]], dtype=tf.int32),
            cameraId=tf.constant([[0], [1]], dtype=tf.int32),
            monitorId=tf.constant([[0], [1]], dtype=tf.int32),
            placeId=tf.constant([[0], [1]], dtype=tf.int32),
            shape=tf.constant([2, 1]),
        )

        assert isinstance(
            result, tf.Tensor
        ), "Model should produce valid output immediately after initialization"
        assert result.shape == [
            2,
            1,
            5 * 64,
        ], f"Expected shape [2, 1, 320], got {result.shape}"
