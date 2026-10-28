"""Multilabel model configurations used by the evaluation layer.

Three families, all wrapped in ``OneVsRestClassifier`` so a multilabel indicator
matrix is handled one binary task per label:

* :class:`LogisticConfig` - ``StandardScaler`` + ``OneVsRest(LogisticRegression)``;
* :class:`RandomForestConfig` - ``OneVsRest(RandomForestClassifier)``, **no**
  scaler (trees are invariant to monotone rescaling);
* :class:`XGBoostConfig` - ``OneVsRest(XGBClassifier)``, **no** scaler,
  **optional**: ``xgboost`` is not a project dependency and the build fails with
  :class:`MissingDependencyError` unless it is installed.

Thresholds are the default 0.5 of ``predict``; no threshold tuning happens here.

Why OvR (Spike S0): ``LogisticRegression``/``RandomForestClassifier`` are binary
or multiclass over *classes*, not over labels, so a multilabel target needs one
binary classifier per label. Alternatives considered: ``MultiOutputClassifier``
(no advantage here), multinomial over label combinations (infeasible with 41
labels), a native multilabel logistic (does not exist).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import ClassVar

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_SOLVERS = ("liblinear", "lbfgs")
SUPPORTED_CLASS_WEIGHTS = (None, "balanced")
SUPPORTED_FOREST_CLASS_WEIGHTS = (None, "balanced", "balanced_subsample")
XGBOOST_MODULE = "xgboost"


class MissingDependencyError(RuntimeError):
    """A model family was requested whose dependency is not installed."""


@dataclass(frozen=True)
class LogisticConfig:
    """Reproducible configuration of the multilabel logistic evaluator."""

    model_id: ClassVar[str] = "logistic_ovr"

    C: float = 1.0
    class_weight: str | None = None
    solver: str = "liblinear"
    max_iter: int = 1000
    tol: float = 1e-4
    n_jobs: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.C, (int, float)) or self.C <= 0:
            raise ValueError(f"C must be > 0; got {self.C!r}.")
        if self.class_weight not in SUPPORTED_CLASS_WEIGHTS:
            raise ValueError(
                f"class_weight must be one of {SUPPORTED_CLASS_WEIGHTS}; got "
                f"{self.class_weight!r}."
            )
        if self.solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"solver must be one of {SUPPORTED_SOLVERS}; got {self.solver!r}."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1; got {self.max_iter!r}.")
        if not isinstance(self.tol, (int, float)) or self.tol <= 0:
            raise ValueError(f"tol must be > 0; got {self.tol!r}.")
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError(f"n_jobs must be a non-zero integer; got {self.n_jobs!r}.")


def build_estimator(
    config: LogisticConfig | "RandomForestConfig" | "XGBoostConfig" | None = None,
) -> Pipeline:
    """Build the pipeline for any supported model configuration.

    ``LogisticConfig`` keeps the original pipeline (``StandardScaler`` +
    ``OneVsRest(LogisticRegression)``) unchanged; the tree families are wrapped
    in ``OneVsRestClassifier`` without any scaler. The scaler, when present,
    lives inside the pipeline so it is fitted on the rows passed to ``fit``.

    Raises:
        MissingDependencyError: When ``XGBoostConfig`` is requested and
            ``xgboost`` is not installed.
        TypeError: For an unsupported configuration object.
    """
    settings = config or LogisticConfig()
    if isinstance(settings, LogisticConfig):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=settings.C,
                            class_weight=settings.class_weight,
                            solver=settings.solver,
                            max_iter=settings.max_iter,
                            tol=settings.tol,
                            random_state=settings.random_state,
                        ),
                        n_jobs=settings.n_jobs,
                    ),
                ),
            ]
        )
    if isinstance(settings, RandomForestConfig):
        return Pipeline(
            [
                (
                    "classifier",
                    OneVsRestClassifier(
                        RandomForestClassifier(**settings.parameters()),
                        n_jobs=settings.n_jobs,
                    ),
                ),
            ]
        )
    if isinstance(settings, XGBoostConfig):
        classifier = _xgboost_classifier()
        return Pipeline(
            [
                (
                    "classifier",
                    OneVsRestClassifier(
                        classifier(**settings.parameters()), n_jobs=settings.n_jobs
                    ),
                ),
            ]
        )
    raise TypeError(
        "Unsupported model configuration: "
        f"{type(settings).__name__}. Expected LogisticConfig, "
        "RandomForestConfig or XGBoostConfig."
    )


def describe_estimator(
    config: LogisticConfig | "RandomForestConfig" | "XGBoostConfig" | None = None,
) -> dict:
    """Parameters recorded in the experiment manifest (one entry per family)."""
    settings = config or LogisticConfig()
    if isinstance(settings, LogisticConfig):
        return {
            "model_id": settings.model_id,
            "type": "OneVsRestClassifier(LogisticRegression)",
            "scaler": "StandardScaler",
            "C": settings.C,
            "class_weight": settings.class_weight,
            "solver": settings.solver,
            "max_iter": settings.max_iter,
            "tol": settings.tol,
            "n_jobs": settings.n_jobs,
            "random_state": settings.random_state,
            "decision_threshold": 0.5,
        }
    if isinstance(settings, RandomForestConfig):
        payload = {
            "model_id": settings.model_id,
            "type": "OneVsRestClassifier(RandomForestClassifier)",
            "scaler": None,
            "decision_threshold": 0.5,
            "ovr_n_jobs": settings.n_jobs,
        }
        payload.update(settings.parameters())
        return payload
    if isinstance(settings, XGBoostConfig):
        payload = {
            "model_id": settings.model_id,
            "type": "OneVsRestClassifier(XGBClassifier)",
            "scaler": None,
            "decision_threshold": 0.5,
            "ovr_n_jobs": settings.n_jobs,
            "dependency": XGBOOST_MODULE,
            "dependency_available": xgboost_available(),
        }
        payload.update(settings.parameters())
        return payload
    raise TypeError(
        f"Unsupported model configuration: {type(settings).__name__}."
    )


def xgboost_available() -> bool:
    """True when the optional ``xgboost`` dependency is importable."""
    return importlib.util.find_spec(XGBOOST_MODULE) is not None


def _xgboost_classifier():
    """Return ``xgboost.XGBClassifier`` or raise a clear configuration error.

    ``xgboost`` is **not** a project dependency; nothing here adds it. Asking for
    :class:`XGBoostConfig` without it must fail loudly instead of silently
    falling back to another model.
    """
    if not xgboost_available():
        raise MissingDependencyError(
            "XGBoostConfig requires the optional 'xgboost' package, which is not "
            "installed. Install it explicitly (for example 'uv add xgboost') or "
            "use LogisticConfig/RandomForestConfig. This project does not add the "
            "dependency automatically."
        )
    from xgboost import XGBClassifier

    return XGBClassifier


def _check_int(name: str, value: object, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return value


def _check_float(name: str, value: object, low: float, high: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number; got {value!r}.")
    if not (low <= float(value) <= high):
        raise ValueError(f"{name} must be in [{low}, {high}]; got {value!r}.")
    return float(value)


@dataclass(frozen=True)
class RandomForestConfig:
    """Configuration of ``OneVsRest(RandomForestClassifier)``.

    No scaler: trees are invariant to monotone feature rescaling, so adding one
    would only cost time. ``n_jobs`` parallelises across labels (one forest per
    label); each forest stays sequential (``n_jobs=1``) to avoid nested
    parallelism. ``random_state`` fixes bootstrapping, so a given configuration
    is reproducible.
    """

    model_id: ClassVar[str] = "random_forest_ovr"

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    class_weight: str | None = None
    n_jobs: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        _check_int("n_estimators", self.n_estimators, 1)
        if self.max_depth is not None:
            _check_int("max_depth", self.max_depth, 1)
        _check_int("min_samples_split", self.min_samples_split, 2)
        _check_int("min_samples_leaf", self.min_samples_leaf, 1)
        if self.max_features not in ("sqrt", "log2", None):
            if isinstance(self.max_features, int) and not isinstance(
                self.max_features, bool
            ):
                _check_int("max_features", self.max_features, 1)
            elif isinstance(self.max_features, float):
                _check_float("max_features", self.max_features, 0.0, 1.0)
            else:
                raise ValueError(
                    "max_features must be 'sqrt', 'log2', an int >= 1, a float "
                    f"in (0, 1] or None; got {self.max_features!r}."
                )
        if self.class_weight not in SUPPORTED_FOREST_CLASS_WEIGHTS:
            raise ValueError(
                f"class_weight must be one of {SUPPORTED_FOREST_CLASS_WEIGHTS}; "
                f"got {self.class_weight!r}."
            )
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError(f"n_jobs must be a non-zero integer; got {self.n_jobs!r}.")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(
                f"random_state must be an integer; got {self.random_state!r}."
            )

    def parameters(self) -> dict:
        """Keyword arguments for ``RandomForestClassifier``."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": 1,
        }


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration of ``OneVsRest(XGBClassifier)`` - optional dependency.

    ``xgboost`` is not installed by this project. Building the estimator without
    it raises :class:`MissingDependencyError`; nothing falls back silently to
    another model. No scaler is applied (boosted trees do not need one).

    ``multi_strategy``/native multilabel trees are deliberately not used: the
    OvR wrapper keeps the label handling identical to the other two families so
    a comparison only varies the model, not the target encoding.
    """

    model_id: ClassVar[str] = "xgboost_ovr"

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    n_jobs: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        _check_int("n_estimators", self.n_estimators, 1)
        _check_int("max_depth", self.max_depth, 1)
        _check_float("learning_rate", self.learning_rate, 1e-6, 1.0)
        _check_float("subsample", self.subsample, 0.0, 1.0)
        _check_float("colsample_bytree", self.colsample_bytree, 0.0, 1.0)
        _check_float("reg_lambda", self.reg_lambda, 0.0, 1e6)
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError(f"n_jobs must be a non-zero integer; got {self.n_jobs!r}.")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(
                f"random_state must be an integer; got {self.random_state!r}."
            )

    def parameters(self) -> dict:
        """Keyword arguments for ``XGBClassifier`` (one binary model per label)."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": 1,
            "eval_metric": "logloss",
            "verbosity": 0,
        }


SUPPORTED_MODEL_CONFIGS = (LogisticConfig, RandomForestConfig, XGBoostConfig)
