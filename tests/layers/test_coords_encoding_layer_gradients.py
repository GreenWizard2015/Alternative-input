"""Gradient-flow tests for CoordsEncodingLayer."""

import tensorflow as tf

from NN.layers.CoordsEncodingLayer import CoordsEncodingLayer


def test_frequency_branch_has_initial_gradient_with_raw_disabled():
    """Frequency/fusion variables should be active on the first train step."""
    coordinates = tf.constant(
        [
            [[0.05, 0.10], [0.25, 0.40], [0.70, 0.20], [0.95, 0.85]],
            [[0.15, 0.80], [0.45, 0.35], [0.60, 0.65], [0.90, 0.05]],
        ],
        dtype=tf.float32,
    )
    layer = CoordsEncodingLayer(
        N=8,
        raw=False,
        max_frequency=7.0,
        final_dropout=0.0,
    )
    _ = layer(coordinates, training=False)

    layer._bottleneck.kernel.assign(tf.eye(layer.config.N))
    layer._fusion_w.assign(
        tf.reshape(
            tf.linspace(0.1, 1.0, tf.size(layer._fusion_w)), layer._fusion_w.shape
        )
    )
    layer._fusion_b.assign(
        tf.reshape(
            tf.linspace(-0.2, 0.2, tf.size(layer._fusion_b)), layer._fusion_b.shape
        )
    )

    tf.debugging.assert_near(
        layer.gates,
        tf.ones_like(layer.gates) * tf.math.tanh(tf.constant(1.0)),
    )

    with tf.GradientTape() as tape:
        output = layer(coordinates, training=False)
        loss_weights = tf.reshape(
            tf.linspace(0.25, 1.75, tf.size(output)),
            tf.shape(output),
        )
        loss = tf.reduce_sum(output * loss_weights)

    variables = [
        layer._fusion_w,
        layer._fusion_b,
        layer._freq_deltas,
        layer._frequency,
    ]
    gradients = tape.gradient(loss, variables)

    for variable, gradient in zip(variables, gradients):
        assert gradient is not None, f"{variable.name} should receive a gradient"
        gradient_norm = tf.reduce_sum(tf.abs(gradient))
        assert (
            float(gradient_norm.numpy()) > 0.0
        ), f"{variable.name} should receive a non-zero gradient"
