from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .phase1_eval import metrics
from .phase1_models import predict_diagonal_gaussian, predict_ncm, predict_tau_path
from .phase1_objects import SufficientStats


@dataclass(frozen=True)
class BaselineResult:
    method_id: str
    pred: np.ndarray
    balanced_accuracy: float
    accuracy: float
    macro_f1: float
    deployable: bool
    notes: str = ""


def _result(method_id: str, y: np.ndarray, pred: np.ndarray, deployable: bool, notes: str = "") -> BaselineResult:
    m = metrics(y, pred)
    return BaselineResult(method_id, pred, m.balanced_accuracy, m.accuracy, m.macro_f1, deployable, notes)


def representation_baselines(stats: SufficientStats, X_test: np.ndarray, y_test: np.ndarray, taus: list[float], alpha0: float) -> tuple[list[BaselineResult], dict[float, np.ndarray]]:
    path = predict_tau_path(stats, X_test, [float(t) for t in taus], float(alpha0))
    out = [
        _result("ncm", y_test, predict_ncm(stats, X_test), True),
        _result("diagonal_class_gaussian", y_test, predict_diagonal_gaussian(stats, X_test, alpha0), True),
        _result("shared_covariance_tau0", y_test, path[0.0], True),
        _result("fixed_tau0_5", y_test, path[0.5], True),
        _result("class_specific_tau1", y_test, path[1.0], True),
    ]
    # Oracle is reported separately as an upper bound and must never be called deployable.
    best_tau = max(sorted(path), key=lambda t: metrics(y_test, path[t]).balanced_accuracy)
    out.append(_result("oracle_best_tau_test_upper_bound", y_test, path[best_tau], False, f"test-label upper bound; tau={best_tau}"))
    return out, path
