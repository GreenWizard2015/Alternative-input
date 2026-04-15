"""Tests for ModelWrapper."""

import tensorflow as tf
from Core.models.ModelWrapper import PredictionOutput
from tests.fixtures.test_inputs import create_test_inputs


class TestModelWrapper:
    """Tests for ModelWrapper gaze prediction model."""

    def test_call_with_valid_inputs(self, cached_model_wrappers, shared_test_inputs):
        """Test orchestrator call method processes inputs correctly."""
        wrapper = cached_model_wrappers["timesteps_2"]
        inputs = shared_test_inputs["small_batch"]

        result = wrapper.call(inputs, training=False)

        assert isinstance(
            result, PredictionOutput
        ), f"Expected PredictionOutput, got {type(result)}"
        assert result.result is not None, "Expected valid result attribute"
        assert result.result.shape == (
            1,
            2,
            2,
        ), f"Expected result shape (1, 2, 2), got {result.result.shape}"

    def test_call_delegates_to_embedding_block(self, cached_model_wrappers):
        """Test that call method properly delegates embedding generation."""
        wrapper = cached_model_wrappers["timesteps_1"]

        # Create test inputs with ones and IDs
        inputs = create_test_inputs(
            batch_size=1,
            timesteps=1,
            input_type="ones",
            include_ids=True,
        )
        # Override IDs with specific values for this test
        inputs["userId"] = tf.constant([[1]], dtype=tf.int32)
        inputs["placeId"] = tf.constant([[2]], dtype=tf.int32)
        inputs["screenId"] = tf.constant([[3]], dtype=tf.int32)
        inputs["cameraId"] = tf.constant([[0]], dtype=tf.int32)
        inputs["monitorId"] = tf.constant([[0]], dtype=tf.int32)

        result = wrapper.call(inputs, training=False)

        assert result.result is not None, "Expected valid result"
        assert result.result.shape == (
            1,
            1,
            2,
        ), f"Expected result shape (1, 1, 2), got {result.result.shape}"

    def test_call_output_values_in_valid_range(self, cached_model_wrappers):
        """Test that call produces reasonable output values."""
        wrapper = cached_model_wrappers["timesteps_2"]

        # Create random normal inputs with ones for time
        inputs = create_test_inputs(
            batch_size=1,
            timesteps=2,
            input_type="random_normal",
            include_ids=True,
        )

        result = wrapper.call(inputs, training=False)

        assert isinstance(
            result.result, tf.Tensor
        ), f"Expected tf.Tensor output, got {type(result.result)}"

        max_magnitude = tf.reduce_max(tf.abs(result.result)).numpy()
        assert (
            max_magnitude < 100.0
        ), f"Output magnitude {max_magnitude} is unreasonable"

    def test_trainable_variables_can_be_used_in_training(self, cached_model_wrappers):
        """Test that trainable_variables can be used in gradient computation."""
        wrapper = cached_model_wrappers["timesteps_2"]

        # Create test inputs with ones
        inputs = create_test_inputs(
            batch_size=1,
            timesteps=2,
            input_type="ones",
            include_ids=True,
        )

        with tf.GradientTape() as tape:
            result = wrapper.call(inputs, training=True)
            loss = tf.reduce_mean(result.result)

        trainable_vars = wrapper.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        non_none_gradients = [g for g in gradients if g is not None]
        assert len(non_none_gradients) > 0, "Expected gradients to be computed"
