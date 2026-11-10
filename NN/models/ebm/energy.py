"""Metric energy and fraction-specific Pair MLP evaluation."""

import tensorflow as tf

from NN.layers.BranchedMLP import BranchedMLP
from NN.models.EBMMetric import create_metric
from NN.models.ebm.config import EBMConfig
from NN.models.ebm.routing import EBMRouting


class EBMEnergy(tf.keras.layers.Layer):
    """Own Pair MLP weights and the common energy calculation for every path."""

    def __init__(self, config: EBMConfig, routing: EBMRouting, **kwargs) -> None:
        kwargs["name"] = kwargs.get("name") or "ebm_energy"
        super().__init__(**kwargs)
        self._config = config
        self._routing = routing
        self._metric = create_metric(config.metric)
        self._energy_pair = None
        self._pair_fraction_vectorized = False

    def build(self, input_shapes: list[tf.TensorShape]) -> None:
        """Build the optional Pair MLP before energy evaluation."""
        if self._config.use_pair_mlp:
            self._energy_pair = BranchedMLP(
                sizes=[1],
                branches=1,
                activation="linear",
                last_activation="tanh",
                use_bias=False,
                use_last_bias=False,
                branched_input=False,
                dropout=0.0,
                name="energy_pair",
            )
            self._energy_pair.build(tf.TensorShape([None, self._config.energy_dim]))
            # For a single-layer Pair MLP, padding is worthwhile only when
            # wasted FLOPs are limited; otherwise use narrow individual passes.
            padded = len(self._config.matryoshka_dims) * self._config.energy_dim
            sliced = sum(self._config.matryoshka_dims)
            self._pair_fraction_vectorized = padded <= 1.75 * sliced
        super().build(input_shapes)

    def _batch_similarity(self, teachers, students):
        # teachers [B,S,N,D], students [B,S,D] -> [B,B,S,N].
        # einsum contracts the embedding dimension, never materializing a [B,B,S,N,D] tensor.
        return tf.einsum("isnd,jsd->ijsn", teachers, students)

    def levels(
        self,
        teachers: tf.Tensor,
        students: tf.Tensor,
        training: bool,
        all_pairs: bool,
    ) -> tf.Tensor:
        """Energy for ALL Matryoshka prefixes, shape [F,B,B,S,N] or [F,B,S,N].

        The pair input is different at each prefix (both the similarity and
        metric-normalized fusion). Supply these inputs as [F,Q,D] to the
        extended BranchedMLP(fractions=..., aligned_fraction_inputs=True)
        when vectorization is cost-effective. Otherwise uses narrow scalar
        passes to avoid full-width padding. Per-prefix similarities remain exact.
        Memory is bounded to one system's fusion at a time.
        """
        sims = []
        for dim in self._config.matryoshka_dims:
            if all_pairs:
                dot = self._batch_similarity(teachers[..., :dim], students[..., :dim])
                sim = self._metric_similarity_batch_systems(
                    teachers, students, dot, dim
                )
            else:
                dot = tf.einsum(
                    "bsnd,bsd->bsn", teachers[..., :dim], students[..., :dim]
                )
                sim = self._metric_similarity_aligned_systems(
                    teachers, students, dot, dim
                )
            sims.append(sim)
        similarities = tf.stack(sims, axis=0)

        mlp = None
        if self._config.use_pair_mlp:
            outputs = []
            for system in range(self._config.num_systems):
                inputs = []
                narrow_results = []
                for index, dim in enumerate(self._config.matryoshka_dims):
                    teacher = teachers[:, system, :, :dim]
                    student = students[:, system, :dim]
                    sim = (
                        sims[index][:, :, system, :]
                        if all_pairs
                        else sims[index][:, system, :]
                    )
                    fusion = (
                        teacher[:, None, :, :]
                        + student[None, :, None, :]
                        + sim[..., None]
                        if all_pairs
                        else teacher + student[:, None, :] + sim[..., None]
                    )
                    fusion = self._metric.prepare(fusion)
                    flat = tf.reshape(fusion, [-1, dim])
                    if self._pair_fraction_vectorized:
                        # Normalize the prefix before padding; otherwise the
                        # cosine fusion changes compared with scalar inference.
                        inputs.append(
                            tf.pad(flat, [[0, 0], [0, self._config.energy_dim - dim]])
                        )
                    else:
                        # Avoid allocating [F,B*B*N,D] when prefixes are tiny.
                        narrow_results.append(
                            self._energy_pair(
                                flat,
                                training=training,
                                fraction=self._config.matryoshka_fractions[index],
                            )[:, 0]
                        )

                if self._pair_fraction_vectorized:
                    # [F,Q,D] -> [F,Q,1]; one multi-fraction call per system.
                    result = self._energy_pair(
                        tf.stack(inputs, axis=0),
                        training=training,
                        fractions=self._config.matryoshka_fractions,
                        aligned_fraction_inputs=True,
                        fraction_strategy="vectorized",
                    )[..., 0]
                else:
                    result = tf.stack(narrow_results, axis=0)

                shape = (
                    tf.concat(
                        [
                            [len(self._config.matryoshka_dims)],
                            tf.shape(sims[0])[:2],
                            [self._config.num_teachers],
                        ],
                        axis=0,
                    )
                    if all_pairs
                    else tf.concat(
                        [
                            [len(self._config.matryoshka_dims)],
                            tf.shape(sims[0])[:1],
                            [self._config.num_teachers],
                        ],
                        axis=0,
                    )
                )
                outputs.append(tf.reshape(result, shape))

            mlp = tf.stack(outputs, axis=3 if all_pairs else 2)
        return self._metric.energy_components(similarities, mlp).energy

    def _metric_similarity_aligned_systems(self, teachers, students, dot, dim):
        return self._metric.system_similarity(teachers, students, dot, dim, False)

    def _metric_similarity_batch_systems(self, teachers, students, dot, dim):
        # Backend expects [B,N,D], [B,S,D] on two batch axes.
        # System-specific teacher tensors are folded into one entity axis.
        # Cosine simply uses the dot; RMSE computes teacher/student norms.
        # This closure is installed by EBMMetric.create_metric.
        return self._metric.system_similarity(teachers, students, dot, dim, True)

    def combine_systems(self, energy: tf.Tensor) -> tf.Tensor:
        """Soft-min across systems in the original energy units."""
        temperature = tf.cast(self._config.system_temperature, tf.float32)
        logits = -tf.cast(energy, tf.float32) / temperature
        return -temperature * (
            self._routing.logsumexp(logits, axis=-2)
            - tf.math.log(tf.cast(self._config.num_systems, tf.float32))
        )

    def cross_system_energy(self, students: tf.Tensor) -> tf.Tensor:
        """Return pair energies between a sample\'s distinct system heads."""
        # [B,S,D] vs [B,S,D] -> [B,S,S]; no BxB or STE graph.
        dot = tf.einsum("btd,bsd->bst", students, students)
        similarity = self._metric.similarity(
            students, students, dot, self._config.energy_dim, False
        )
        mlp = None
        if self._config.use_pair_mlp:
            fusion = self._metric.prepare(
                students[:, None, :, :]
                + students[:, :, None, :]
                + similarity[..., None]
            )
            mlp = tf.reshape(
                self._energy_pair(
                    tf.reshape(fusion, [-1, self._config.energy_dim]), training=True
                ),
                tf.shape(similarity),
            )
        cross = self._metric.energy_components(similarity, mlp).energy
        return cross
