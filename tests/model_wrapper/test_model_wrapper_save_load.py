"""Tests for ModelWrapper save/load functionality."""

import tempfile
import tensorflow as tf
import numpy as np

from Core.models.ModelWrapper import ModelWrapper
from tests.fixtures.embedding_fixtures import MODEL_CONFIGS


class TestModelWrapperSaveLoad:
    """Tests for ModelWrapper save and load methods."""

    def test_save_load_with_postfix_preserves_predictions(
        self, stats_data, shared_test_inputs
    ):
        """Test that save/load with custom postfix works correctly."""
        config = MODEL_CONFIGS["fast"]

        wrapper1 = ModelWrapper(
            timesteps=config["timesteps"],
            stats=stats_data,
            embeddingSize=config["embeddingSize"],
        )

        inputs = shared_test_inputs["small_batch"]
        prediction_before = wrapper1.call(inputs, training=False).result
        weights_before = wrapper1.weights_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper1.save(tmpdir, postfix="best")

            wrapper2 = ModelWrapper(
                timesteps=config["timesteps"],
                stats=stats_data,
                embeddingSize=config["embeddingSize"],
            )
            weights_after = wrapper2.weights_dict()
            assert len(weights_before.keys()) == len(
                weights_after.keys()
            ), "Weight keys differ number"
            wrapper2.load(tmpdir, postfix="best")

            prediction_after = wrapper2.call(inputs, training=False).result

            # Check that keys are identical
            keys_before = set(weights_before.keys())
            keys_after = set(weights_after.keys())
            key_diff = keys_before.symmetric_difference(keys_after)

            assert not key_diff, f"Weight keys differ: {key_diff}"
            # also check weights same after load
            weights_after = wrapper2.weights_dict()
            for k, weights_a in weights_before.items():
                weights_b = weights_after[k]
                assert np.array_equal(weights_a, weights_b)

            assert (
                prediction_after.shape == prediction_before.shape
            ), f"Output shape mismatch: {prediction_after.shape} vs {prediction_before.shape}"

            max_diff = tf.reduce_max(
                tf.abs(prediction_after - prediction_before)
            ).numpy()
            assert max_diff < 1e-5, f"Predictions diverged: max_diff = {max_diff:.2e}"
