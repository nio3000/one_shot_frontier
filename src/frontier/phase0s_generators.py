from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .phase0h_generators import (
    Phase0HGeometry,
    build_phase0h_geometry,
    prefixes_to_xy,
    stable_seed,
)

COMMUTING = "mean_preserving_shared_eigenvector_eigenvalue_heterogeneity"
NONCOMMUTING = "mean_preserving_noncommuting_covariance_perturbation"


def panel_geometry(
    panel: str,
    *,
    C: int,
    d: int,
    q_ratio: float,
    target_h: float,
    seed: int,
    mean_scale: float,
) -> Phase0HGeometry:
    if panel in {"panel_a", "panel_c", "smoke"}:
        family = COMMUTING
    elif panel == "panel_b":
        family = NONCOMMUTING
    else:
        raise ValueError(f"Unknown Phase 0-S panel: {panel}")
    return build_phase0h_geometry(
        generator_family=family,
        C=C,
        d=d,
        q_ratio=q_ratio,
        target_h=target_h,
        seed=seed,
        mean_scale=mean_scale,
        reff_tolerance=1e-6,
    )


def n_per_class_from_seff(seff: float, actual_reff: float) -> int:
    return max(3, int(np.ceil(float(seff) * float(actual_reff))) + 1)


def _sample_gaussian_class(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
    return rng.multivariate_normal(mean, cov, size=int(n), check_valid="raise")


def _sample_student_t_class(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    n: int,
    nu: float,
) -> np.ndarray:
    # z~N(0,cov), u~chi2_nu, x = mean + sqrt((nu-2)/u) z => Cov(x)=cov for nu>2.
    z = rng.multivariate_normal(np.zeros(len(mean)), cov, size=int(n), check_valid="raise")
    u = rng.chisquare(float(nu), size=int(n))
    scale = np.sqrt((float(nu) - 2.0) / u)[:, None]
    return mean[None, :] + scale * z


def sample_xy(
    geom: Phase0HGeometry,
    *,
    n_per_class: int,
    seed: int,
    distribution: str,
    nu: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    xs, ys = [], []
    for c in range(geom.C):
        if distribution == "gaussian":
            x = _sample_gaussian_class(rng, geom.means[c], geom.covs[c], n_per_class)
        elif distribution == "student_t":
            x = _sample_student_t_class(rng, geom.means[c], geom.covs[c], n_per_class, nu)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        xs.append(x)
        ys.append(np.full(int(n_per_class), c, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


def sample_balanced_test(
    geom: Phase0HGeometry,
    *,
    n_total: int,
    seed: int,
    distribution: str,
    nu: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    per = max(1, int(n_total) // geom.C)
    return sample_xy(geom, n_per_class=per, seed=seed, distribution=distribution, nu=nu)
