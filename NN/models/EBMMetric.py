"""Cosine, bounded-RMSE and additive cosine+RMSE backends for ContrastiveEBM.

``create_metric`` returns MetricFunctions NamedTuple with seven callables:
    prepare(x) -> processed vectors
    similarity(teachers, students, dot, dim, all_pairs) -> similarity/score
    energy_components(similarity, mlp=None) -> (energy, reconstruction, mlp_component)
    reference_energy(reference, teachers, dim) -> [B,B,N]
    repulsion(grouped_latents) -> scalar
    alignment(student, target) -> per-sample loss [B]
    system_similarity(teachers [B,S,N,D], students [B,S,D], dot, dim, all_pairs)

Cosine/RMSE energies are in [0,2]. Additive cosine+rmse is in [0,4].
Thresholds for the additive metric sum both original calibration thresholds.
Cosine inside the composite is normalized PER Matryoshka prefix while
projected features use the bounded RMSE preparation, never L2 preparation.
General EBM hyperparameters remain in ContrastiveEBM.__init__.
The selection of cosine vs RMSE is made once in Python at model construction;
no metric-dependent Python branches are required in tf.function training paths.
"""

from typing import Callable, NamedTuple, Optional

import tensorflow as tf


class MetricDefaults(NamedTuple):
    """Properties of a similarity/energy backend, not EBM training settings."""

    energy_min: float
    energy_max: float
    positive_target: float
    negative_margin: float
    rmse_feature_limit: Optional[float] = None
    rmse_repulsion_margin: Optional[float] = None


class EnergyComponents(NamedTuple):
    """Decomposition of a metric Energy into TensorFlow tensors."""

    energy: tf.Tensor
    reconstruction: tf.Tensor
    mlp_component: tf.Tensor


class MetricFunctions(NamedTuple):
    """Metric backend operations; tuple unpacking remains backward-compatible."""

    prepare: Callable
    similarity: Callable
    energy_components: Callable
    reference_energy: Callable
    repulsion: Callable
    alignment: Callable
    system_similarity: Callable


METRIC_DEFAULTS = {
    "cosine": MetricDefaults(
        energy_min=0.0, energy_max=2.0, positive_target=0.2, negative_margin=0.8
    ),
    "rmse": MetricDefaults(
        energy_min=0.0,
        energy_max=2.0,
        positive_target=0.2,
        negative_margin=0.8,
        rmse_feature_limit=1.0,
        rmse_repulsion_margin=0.5,
    ),
    "cosine+rmse": MetricDefaults(
        energy_min=0.0,
        energy_max=4.0,
        positive_target=0.4,
        negative_margin=1.6,
        rmse_feature_limit=1.0,
        rmse_repulsion_margin=0.5,
    ),
}


def get_metric_defaults(metric: str) -> MetricDefaults:
    metric = {
        "cos": "cosine",
        "cos+rmse": "cosine+rmse",
        "cos_rmse": "cosine+rmse",
    }.get(metric, metric)
    try:
        return METRIC_DEFAULTS[metric]
    except KeyError as exc:
        raise ValueError("metric must be 'cosine', 'rmse', or 'cosine+rmse'.") from exc


def create_metric(metric: str) -> MetricFunctions:
    """Return named backend operations via MetricFunctions.

    Metric-specific energy ranges and calibration thresholds live in METRIC_DEFAULTS.
    """
    defaults = get_metric_defaults(metric)
    metric = {
        "cos": "cosine",
        "cos+rmse": "cosine+rmse",
        "cos_rmse": "cosine+rmse",
    }.get(metric, metric)
    rmse_feature_limit = defaults.rmse_feature_limit
    rmse_repulsion_margin = defaults.rmse_repulsion_margin
    if metric in ("rmse", "cosine+rmse") and (
        rmse_feature_limit is None
        or rmse_feature_limit <= 0.0
        or rmse_repulsion_margin is None
        or rmse_repulsion_margin <= 0.0
    ):
        raise ValueError("RMSE feature_limit and repulsion_margin must be > 0.")
    if (
        not defaults.energy_min
        <= defaults.positive_target
        < defaults.negative_margin
        <= defaults.energy_max
    ):
        raise ValueError(
            f"Expected {defaults.energy_min} <= positive_target < negative_margin <= {defaults.energy_max} for {metric}."
        )

    if metric == "cosine+rmse":
        # Bounded per-coordinate preparation, identical to RMSE. Never L2
        # normalize feature vectors here: Euclidean distances need magnitude.
        limit = float(rmse_feature_limit)
        margin = float(rmse_repulsion_margin * limit)

        def prepare(x):
            feature_limit = tf.cast(limit, x.dtype)
            return feature_limit * (x / (feature_limit + tf.abs(x)))

        def combined_similarity(teacher_sq, student_sq, dot, dim):
            # Compute both distances from the SAME dot products. Cosine is
            # normalized independently for each Matryoshka prefix.
            eps = tf.cast(1e-12, dot.dtype)
            cosine = tf.math.divide_no_nan(
                dot, tf.sqrt(tf.maximum(teacher_sq * student_sq, eps))
            )
            cosine = tf.clip_by_value(cosine, -1.0, 1.0)
            squared = tf.maximum(teacher_sq + student_sq - 2.0 * dot, 0.0)
            # Positive epsilon avoids a singular sqrt derivative at equality.
            # Subtract its floor, preserving E_RMSE=0 for identical vectors.
            epsilon = tf.cast(1e-6, dot.dtype)
            rmse = tf.maximum(
                tf.sqrt(squared / tf.cast(dim, dot.dtype) + epsilon) - tf.sqrt(epsilon),
                0.0,
            )
            return cosine - rmse / tf.cast(limit, dot.dtype)

        def similarity(teachers, students, dot, dim, all_pairs):
            teacher_sq = tf.reduce_sum(tf.square(teachers[..., :dim]), axis=-1)
            student_sq = tf.reduce_sum(tf.square(students[..., :dim]), axis=-1)
            if all_pairs:
                teacher_sq = teacher_sq[:, None, None, :]
                student_sq = student_sq[None, :, :, None]
            else:
                teacher_sq = teacher_sq[:, None, :]
                student_sq = student_sq[:, :, None]
            return combined_similarity(teacher_sq, student_sq, dot, dim)

        def system_similarity(teachers, students, dot, dim, all_pairs):
            teacher_sq = tf.reduce_sum(tf.square(teachers[..., :dim]), axis=-1)
            student_sq = tf.reduce_sum(tf.square(students[..., :dim]), axis=-1)
            if all_pairs:
                teacher_sq = teacher_sq[:, None, :, :]
                student_sq = student_sq[None, :, :, None]
            else:
                student_sq = student_sq[:, :, None]
            return combined_similarity(teacher_sq, student_sq, dot, dim)

        def energy_components(sim, mlp=None):
            # sim = cosine - normalized_euclidean_RMSE, so this is EXACTLY:
            # (1 - cosine) + normalized_euclidean_RMSE, WITHOUT averaging.
            reconstruction = 1.0 - sim
            mlp_component = (
                0.5 * mlp if mlp is not None else tf.zeros_like(reconstruction)
            )
            raw = reconstruction + mlp_component
            clipped = tf.clip_by_value(
                raw, tf.cast(0.0, raw.dtype), tf.cast(4.0, raw.dtype)
            )
            return EnergyComponents(
                energy=raw + tf.stop_gradient(clipped - raw),
                reconstruction=reconstruction,
                mlp_component=mlp_component,
            )

        def reference_energy(reference, teachers, dim):
            a = reference[..., :dim]
            t = teachers[..., :dim]
            dot = tf.einsum("id,jnd->ijn", a, t)
            anchor_sq = tf.reduce_sum(tf.square(a), axis=-1)[:, None, None]
            teacher_sq = tf.reduce_sum(tf.square(t), axis=-1)[None, :, :]
            return energy_components(
                combined_similarity(anchor_sq, teacher_sq, dot, dim)
            ).energy

        def repulsion(latents):
            # Same bounded Euclidean repulsion as the RMSE backend.
            latents = tf.cast(latents, tf.float32)
            batch = tf.shape(latents)[1]
            dim = tf.cast(tf.shape(latents)[-1], latents.dtype)
            norms = tf.reduce_mean(tf.square(latents), axis=-1)
            distances2 = tf.maximum(
                norms[:, :, None]
                + norms[:, None, :]
                - 2.0 * tf.matmul(latents, latents, transpose_b=True) / dim,
                0.0,
            )
            distances = tf.sqrt(distances2 + 1e-8)
            penalty = tf.square(
                tf.nn.relu(tf.cast(margin, distances.dtype) - distances)
            )
            mask = 1.0 - tf.eye(batch, dtype=penalty.dtype)
            count = tf.cast(tf.shape(latents)[0] * batch * (batch - 1), penalty.dtype)
            return tf.math.divide_no_nan(
                tf.reduce_sum(penalty * mask[None, :, :]), count
            )

        def alignment(student, reference):
            return tf.reduce_mean(tf.square(student - reference), axis=-1)

        return MetricFunctions(
            prepare=prepare,
            similarity=similarity,
            energy_components=energy_components,
            reference_energy=reference_energy,
            repulsion=repulsion,
            alignment=alignment,
            system_similarity=system_similarity,
        )

    if metric == "cosine":

        def prepare(x):
            return tf.math.l2_normalize(x, axis=-1)

        def similarity(teachers, students, dot, dim, all_pairs):
            # Vectors are normalized at the FULL dimension, not per Matryoshka prefix.
            return dot

        def energy_components(sim, mlp=None):
            reconstruction = 1.0 - (0.5 * sim if mlp is not None else sim)
            mlp_component = (
                0.5 * mlp if mlp is not None else tf.zeros_like(reconstruction)
            )
            raw = reconstruction + mlp_component
            clipped = tf.clip_by_value(
                raw, tf.cast(0.0, raw.dtype), tf.cast(2.0, raw.dtype)
            )
            return EnergyComponents(
                energy=raw + tf.stop_gradient(clipped - raw),
                reconstruction=reconstruction,
                mlp_component=mlp_component,
            )

        def reference_energy(reference, teachers, dim):
            return 1.0 - tf.einsum(
                "id,jnd->ijn", reference[..., :dim], teachers[..., :dim]
            )

        def repulsion(latents):
            batch = tf.shape(latents)[1]
            dot = tf.nn.relu(tf.matmul(latents, latents, transpose_b=True))
            count = tf.cast(tf.shape(latents)[0] * batch * (batch - 1), dot.dtype)
            return tf.math.divide_no_nan(
                tf.reduce_sum(dot) - tf.reduce_sum(tf.linalg.diag_part(dot)), count
            )

        def alignment(student, reference):
            # Inputs already have unit L2 norm. Equivalent to 1 - cosine up to
            # floating-point rounding; MSE form is nonnegative by construction.
            return 0.5 * tf.reduce_sum(tf.square(student - reference), axis=-1)

        def system_similarity(teachers, students, dot, dim, all_pairs):
            return dot

        return MetricFunctions(
            prepare=prepare,
            similarity=similarity,
            energy_components=energy_components,
            reference_energy=reference_energy,
            repulsion=repulsion,
            alignment=alignment,
            system_similarity=system_similarity,
        )

    limit = float(rmse_feature_limit)
    margin = float(rmse_repulsion_margin * rmse_feature_limit)

    def prepare(x):
        # Smooth per-coordinate bound, NOT L2 normalization.
        feature_limit = tf.cast(limit, x.dtype)
        return feature_limit * (x / (feature_limit + tf.abs(x)))

    def similarity(teachers, students, dot, dim, all_pairs):
        teacher_sq = tf.reduce_sum(tf.square(teachers[..., :dim]), axis=-1)
        student_sq = tf.reduce_sum(tf.square(students[..., :dim]), axis=-1)
        squared = (
            teacher_sq[:, None, None, :] + student_sq[None, :, :, None] - 2.0 * dot
            if all_pairs
            else teacher_sq[:, None, :] + student_sq[:, :, None] - 2.0 * dot
        )
        rmse = tf.sqrt(
            tf.maximum(squared / tf.cast(dim, dot.dtype), 0.0)
            + tf.cast(1e-8, dot.dtype)
        )
        return 1.0 - rmse / tf.cast(limit, dot.dtype)

    def energy_components(sim, mlp=None):
        # sim = 1 - RMSE / feature_limit in [-1, 1].
        # With Pair MLP the affine re-normalization is EXACTLY
        # E_new = (E_old + 16) / 16 = 1 - 0.5*sim + 0.5*mlp.
        # Without Pair MLP, use E=RMSE/feature_limit in [0, 2] so
        # identical vectors can achieve energy zero.
        reconstruction = 1.0 - (0.5 * sim if mlp is not None else sim)
        mlp_component = 0.5 * mlp if mlp is not None else tf.zeros_like(reconstruction)
        raw = reconstruction + mlp_component
        clipped = tf.clip_by_value(
            raw, tf.cast(0.0, raw.dtype), tf.cast(2.0, raw.dtype)
        )
        return EnergyComponents(
            energy=raw + tf.stop_gradient(clipped - raw),
            reconstruction=reconstruction,
            mlp_component=mlp_component,
        )

    def reference_energy(reference, teachers, dim):
        anchor = reference[..., :dim]
        projected_teachers = teachers[..., :dim]
        dot = tf.einsum("id,jnd->ijn", anchor, projected_teachers)
        anchor_sq = tf.reduce_sum(tf.square(anchor), axis=-1)
        teacher_sq = tf.reduce_sum(tf.square(projected_teachers), axis=-1)
        squared = tf.maximum(
            anchor_sq[:, None, None] + teacher_sq[None, :, :] - 2.0 * dot, 0.0
        )
        rmse = tf.sqrt(squared / tf.cast(dim, dot.dtype) + tf.cast(1e-8, dot.dtype))
        # Reference-teacher uses the metric's base distance (no Pair MLP).
        return tf.clip_by_value(
            rmse / tf.cast(limit, dot.dtype),
            tf.cast(0.0, dot.dtype),
            tf.cast(2.0, dot.dtype),
        )

    def repulsion(latents):
        # Bounded distance-margin repulsion; does not reward exploding norms.
        latents = tf.cast(latents, tf.float32)
        batch = tf.shape(latents)[1]
        dim = tf.cast(tf.shape(latents)[-1], latents.dtype)
        norms = tf.reduce_mean(tf.square(latents), axis=-1)
        distances2 = tf.maximum(
            norms[:, :, None]
            + norms[:, None, :]
            - 2.0 * tf.matmul(latents, latents, transpose_b=True) / dim,
            0.0,
        )
        distances = tf.sqrt(distances2 + 1e-8)
        penalty = tf.square(tf.nn.relu(tf.cast(margin, distances.dtype) - distances))
        mask = 1.0 - tf.eye(batch, dtype=penalty.dtype)
        count = tf.cast(tf.shape(latents)[0] * batch * (batch - 1), penalty.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(penalty * mask[None, :, :]), count)

    def alignment(student, reference):
        # Bounded-RMSE vectors. Squared per-coordinate error is scale-stable.
        return tf.reduce_mean(tf.square(student - reference), axis=-1)

    def system_similarity(teachers, students, dot, dim, all_pairs):
        t_sq = tf.reduce_sum(tf.square(teachers[..., :dim]), axis=-1)
        s_sq = tf.reduce_sum(tf.square(students[..., :dim]), axis=-1)
        distances2 = (
            t_sq[:, None, :, :] + s_sq[None, :, :, None] - 2.0 * dot
            if all_pairs
            else t_sq + s_sq[:, :, None] - 2.0 * dot
        )
        rmse = tf.sqrt(tf.maximum(distances2 / tf.cast(dim, dot.dtype), 0.0) + 1e-8)
        return 1.0 - rmse / tf.cast(limit, dot.dtype)

    return MetricFunctions(
        prepare=prepare,
        similarity=similarity,
        energy_components=energy_components,
        reference_energy=reference_energy,
        repulsion=repulsion,
        alignment=alignment,
        system_similarity=system_similarity,
    )
