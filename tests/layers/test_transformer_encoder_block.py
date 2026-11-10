"""Tests for TransformerEncoderBlock."""

import tensorflow as tf

from NN.layers.TransformerEncoderBlock import TransformerEncoderBlock


def test_prime_d_model_falls_back_to_single_attention_head():
    """Prime d_model should not leave num_heads at zero."""
    block = TransformerEncoderBlock(d_model=17, max_dim=8, dropout_rate=0.0)
    x = tf.random.normal((2, 3, 17))

    y = block(x, training=False)

    assert block.num_heads == 1
    assert y.shape == x.shape
