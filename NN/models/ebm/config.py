"""Immutable EBM architecture and training configuration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from NN.models.EBMMetric import get_metric_defaults


EBM_SETTINGS_DEFAULTS = {
    "system_temperature": 0.1,
    "balance_weight": 0.1,
    "calibration_weight": 0.01,
    "routing_entropy_weight": 0.0,
    "reference_teacher_weight": 0.5,
    "ste_training_weight": 0.1,
    "alignment_weight": 0.1,
    "cross_system_weight": 0.1,
}


def _resolve_ebm_settings(settings: Mapping | None) -> dict[str, float]:
    """Merge partial overrides with defaults, rejecting misspelled keys."""
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping (dict) or None.")
    unknown = set(settings) - set(EBM_SETTINGS_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown EBM settings: {', '.join(sorted(map(str, unknown)))}"
        )
    resolved = {**EBM_SETTINGS_DEFAULTS, **settings}
    if resolved["system_temperature"] <= 0.0:
        raise ValueError("system_temperature must be > 0.")
    if any(
        resolved[key] < 0.0
        for key in (
            "balance_weight",
            "calibration_weight",
            "routing_entropy_weight",
            "reference_teacher_weight",
            "ste_training_weight",
            "alignment_weight",
            "cross_system_weight",
        )
    ):
        raise ValueError("Loss weights must be nonnegative.")
    return resolved


@dataclass(frozen=True)
class EBMConfig:
    """Settings shared by encoding, energy, routing and objectives."""

    num_systems: int = 4
    matryoshka_n: int | None = None
    energy_dim: int | None = None
    use_pair_mlp: bool = False
    metric: str = "cosine+rmse"
    system_temperature: float = 0.1
    balance_weight: float = 0.1
    calibration_weight: float = 0.01
    routing_entropy_weight: float = 0.0
    reference_teacher_weight: float = 0.5
    ste_training_weight: float = 0.1
    alignment_weight: float = 0.1
    cross_system_weight: float = 0.1
    student_dim: int | None = None
    teacher_dims: tuple[int, ...] = ()
    matryoshka_dims: tuple[int, ...] = ()

    @classmethod
    def create(
        cls,
        num_systems: int,
        matryoshka_n: int | None,
        energy_dim: int | None,
        use_pair_mlp: bool,
        metric: str,
        settings: Mapping | None,
    ) -> "EBMConfig":
        """Validate architecture options and merge partial training settings."""
        metric = {
            "cos": "cosine",
            "cos+rmse": "cosine+rmse",
            "cos_rmse": "cosine+rmse",
        }.get(metric, metric)
        if num_systems < 1:
            raise ValueError("num_systems must be >= 1.")
        if matryoshka_n is not None and matryoshka_n < 2:
            raise ValueError("matryoshka_n must be >= 2 or None.")
        if energy_dim is not None and energy_dim < 1:
            raise ValueError("energy_dim must be >= 1 or None.")
        get_metric_defaults(metric)
        resolved = _resolve_ebm_settings(settings)
        return cls(
            num_systems=int(num_systems),
            matryoshka_n=None if matryoshka_n is None else int(matryoshka_n),
            energy_dim=None if energy_dim is None else int(energy_dim),
            use_pair_mlp=bool(use_pair_mlp),
            metric=metric,
            **{key: float(value) for key, value in resolved.items()},
        )

    def with_dimensions(
        self,
        student_dim: int,
        teacher_dims: tuple[int, ...],
    ) -> "EBMConfig":
        """Resolve projection width and Matryoshka prefixes after input validation."""
        energy_dim = self.energy_dim
        if energy_dim is None:
            energy_dim = 4 * math.ceil(2 * max(1.0, math.sqrt(student_dim)) / 4)
        dims = []
        if self.matryoshka_n is not None:
            dim = min(self.matryoshka_n, energy_dim)
            while dim < energy_dim:
                dims.append(dim)
                dim *= self.matryoshka_n
        dims.append(energy_dim)
        return replace(
            self,
            student_dim=student_dim,
            teacher_dims=teacher_dims,
            energy_dim=energy_dim,
            matryoshka_dims=tuple(dims),
        )

    @property
    def num_teachers(self) -> int:
        """Number of teacher inputs."""
        return len(self.teacher_dims)

    @property
    def matryoshka_fractions(self) -> tuple[float, ...]:
        """Projection fractions in prefix order."""
        if self.energy_dim is None:
            return ()
        return tuple(dim / self.energy_dim for dim in self.matryoshka_dims)
