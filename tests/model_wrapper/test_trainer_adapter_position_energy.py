"""Tests for TrainerAdapter better-latent energy composition."""

import tensorflow as tf

from Core.models.TrainerAdapter import TrainerAdapter


class CountingPositionEnergy:
    def __init__(self):
        self.calls = 0

    def __call__(self, features, training=False):
        self.calls += 1
        _, z = features
        energy = tf.reduce_sum(tf.square(z), axis=-1, keepdims=True)
        return energy, features


def test_collect_better_latents_uses_position_energy():
    adapter = TrainerAdapter("test")
    adapter._create_if_needed()
    adapter._pos_nograd = CountingPositionEnergy()

    batch_size = 2
    timesteps = 2
    latent_dim = 64
    student_latents = tf.random.normal((batch_size, timesteps, latent_dim))
    teachers = [
        tf.random.normal((batch_size, timesteps, latent_dim)),
        tf.random.normal((batch_size, timesteps, latent_dim)),
    ]
    loss_weights = tf.ones((batch_size, timesteps, len(teachers)))
    y_true = tf.random.normal((batch_size, timesteps, 2))
    own_advantage = tf.ones((batch_size, timesteps, 1))

    better_latents, init_energy = adapter._collect_better_latents(
        student_latents,
        teachers,
        loss_weights,
        y_true=y_true,
        own_advantage=own_advantage,
    )

    assert adapter._pos_nograd.calls > 0
    assert init_energy.shape == (batch_size, timesteps, 1)
    assert len(better_latents) == 3
    for item in better_latents:
        assert item["z"].shape == student_latents.shape
        assert item["energy"].shape == (batch_size, timesteps, 1)
