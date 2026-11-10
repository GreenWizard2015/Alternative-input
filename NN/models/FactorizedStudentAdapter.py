import tensorflow as tf


class FactorizedStudentAdapter(tf.keras.layers.Layer):
    def __init__(
        self, num_targets, num_factors, factor_dim, normalize, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._num_targets = num_targets
        self._num_factors = num_factors
        self._factor_dim = factor_dim
        self._num_heads = num_targets * num_factors
        self._normalize = tf.keras.layers.Lambda(normalize)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if input_dim is None:
            raise ValueError("Student latent dimension must be statically known.")

        input_dim = int(input_dim)
        hidden_dim = self._num_heads * self._factor_dim
        first_limit = (6.0 / (input_dim + self._factor_dim)) ** 0.5
        second_limit = (3.0 / self._factor_dim) ** 0.5

        self._w1 = self.add_weight(
            name="w1",
            shape=(input_dim, hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(-first_limit, first_limit),
            trainable=True,
        )
        self._b1 = self.add_weight(
            name="b1", shape=(hidden_dim,), initializer="zeros", trainable=True
        )
        self._w2 = self.add_weight(
            name="w2",
            shape=(self._num_heads, self._factor_dim, self._factor_dim),
            initializer=tf.keras.initializers.RandomUniform(
                -second_limit, second_limit
            ),
            trainable=True,
        )
        self._b2 = self.add_weight(
            name="b2",
            shape=(self._num_heads, self._factor_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, student, training=False):
        batch_size = tf.shape(student)[0]

        hidden = tf.matmul(student, self._w1) + self._b1
        hidden = tf.nn.relu(hidden)
        hidden = tf.reshape(hidden, (batch_size, self._num_heads, self._factor_dim))

        factors = tf.einsum("bhr,hru->bhu", hidden, self._w2) + self._b2[None]
        factors = tf.reshape(
            factors,
            (batch_size, self._num_targets, self._num_factors * self._factor_dim),
        )
        factors = self._normalize(factors)
        factors = tf.reshape(
            factors,
            (batch_size, self._num_targets, self._num_factors, self._factor_dim),
        )
        factors = tf.transpose(factors, (1, 0, 2, 3))

        tf.debugging.assert_equal(
            tf.shape(factors),
            (self._num_targets, batch_size, self._num_factors, self._factor_dim),
        )
        tf.debugging.assert_all_finite(factors, "Non-finite student factors.")

        return factors
