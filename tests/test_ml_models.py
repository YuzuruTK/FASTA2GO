"""Tests for the multilabel model configurations (synthetic data only)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protein_database.ml import models
from protein_database.ml.models import (
    SUPPORTED_MODEL_CONFIGS,
    LogisticConfig,
    MissingDependencyError,
    RandomForestConfig,
    XGBoostConfig,
    build_estimator,
    describe_estimator,
    xgboost_available,
)


def test_estimator_is_scaler_plus_one_vs_rest() -> None:
    pipeline = build_estimator()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], OneVsRestClassifier)


def test_describe_estimator_records_every_parameter() -> None:
    payload = describe_estimator(LogisticConfig(C=0.5, class_weight="balanced"))
    assert payload["type"] == "OneVsRestClassifier(LogisticRegression)"
    assert payload["scaler"] == "StandardScaler"
    assert payload["C"] == 0.5
    assert payload["class_weight"] == "balanced"
    assert payload["decision_threshold"] == 0.5
    assert payload["random_state"] == 42


def test_estimator_fits_and_predicts_multilabel_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.column_stack([X[:, 0] > 0, X[:, 1] > 0])
    pipeline = build_estimator(LogisticConfig(max_iter=200))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == y.shape
    # OvR returns a 0/1 indicator; the evaluation layer converts it to bool
    assert set(np.unique(predictions)) <= {0, 1}
    # the scaler was fitted inside the pipeline, on the rows passed to fit
    assert hasattr(pipeline.named_steps["scaler"], "mean_")


@pytest.mark.parametrize(
    "override",
    [
        {"C": 0},
        {"C": -1.0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"class_weight": "something"},
        {"solver": "saga"},
        {"n_jobs": 0},
    ],
)
def test_config_rejects_invalid_values(override: dict) -> None:
    with pytest.raises(ValueError):
        LogisticConfig(**override)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def test_random_forest_is_one_vs_rest_without_a_scaler() -> None:
    pipeline = build_estimator(RandomForestConfig(n_estimators=5))
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["classifier"]
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, OneVsRestClassifier)
    assert isinstance(classifier.estimator, RandomForestClassifier)
    # trees do not need scaling, so no scaler may sneak into the pipeline
    assert "scaler" not in pipeline.named_steps


def test_random_forest_forwards_its_parameters() -> None:
    config = RandomForestConfig(
        n_estimators=7,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="log2",
        class_weight="balanced_subsample",
        random_state=11,
    )
    forest = build_estimator(config).named_steps["classifier"].estimator
    assert forest.n_estimators == 7
    assert forest.max_depth == 3
    assert forest.min_samples_split == 4
    assert forest.min_samples_leaf == 2
    assert forest.max_features == "log2"
    assert forest.class_weight == "balanced_subsample"
    assert forest.random_state == 11
    # the forest itself stays sequential; OvR owns the parallelism
    assert forest.n_jobs == 1


def test_random_forest_describe_records_no_scaler() -> None:
    payload = describe_estimator(RandomForestConfig(n_estimators=3, n_jobs=2))
    assert payload["model_id"] == "random_forest_ovr"
    assert payload["type"] == "OneVsRestClassifier(RandomForestClassifier)"
    assert payload["scaler"] is None
    assert payload["n_estimators"] == 3
    assert payload["ovr_n_jobs"] == 2
    assert payload["decision_threshold"] == 0.5


def test_random_forest_is_deterministic_with_a_fixed_random_state() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = np.column_stack([X[:, 0] > 0, X[:, 1] > 0]).astype(bool)
    config = RandomForestConfig(n_estimators=10, random_state=7)
    first = build_estimator(config).fit(X, y).predict(X)
    second = build_estimator(config).fit(X, y).predict(X)
    assert np.array_equal(first, second)


def test_random_forest_handles_multilabel_with_a_rare_label() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    rare = np.zeros(60, dtype=bool)
    rare[[0, 5]] = True
    y = np.column_stack([X[:, 0] > 0, rare])
    predictions = build_estimator(RandomForestConfig(n_estimators=5)).fit(X, y).predict(X)
    assert predictions.shape == y.shape


@pytest.mark.parametrize(
    "override",
    [
        {"n_estimators": 0},
        {"max_depth": 0},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"max_features": "auto"},
        {"class_weight": "unknown"},
        {"n_jobs": 0},
        {"random_state": "seed"},
    ],
)
def test_random_forest_config_rejects_invalid_values(override: dict) -> None:
    with pytest.raises(ValueError):
        RandomForestConfig(**override)




# ---------------------------------------------------------------------------
# XGBoost (optional dependency: present or absent, never simulated as quality)
# ---------------------------------------------------------------------------


class _RecordingBooster(ClassifierMixin, BaseEstimator):
    """Stand-in for ``XGBClassifier`` used only to test the wiring.

    It is *not* XGBoost: it records the keyword arguments it was built with and
    returns a constant prediction. No performance claim is made from it.
    """

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return dict(self.kwargs)

    def set_params(self, **params: object) -> "_RecordingBooster":
        self.kwargs.update(params)
        return self

    def fit(self, X, y):  # noqa: N803 - sklearn signature
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):  # noqa: N803 - sklearn signature
        return np.zeros(X.shape[0], dtype=int)


def test_xgboost_availability_matches_the_environment() -> None:
    expected = importlib.util.find_spec("xgboost") is not None
    assert xgboost_available() is expected


def test_missing_xgboost_raises_a_clear_configuration_error(monkeypatch) -> None:
    monkeypatch.setattr(models, "xgboost_available", lambda: False)
    with pytest.raises(MissingDependencyError, match="xgboost"):
        build_estimator(XGBoostConfig(n_estimators=5))


def test_missing_xgboost_does_not_break_other_models(monkeypatch) -> None:
    monkeypatch.setattr(models, "xgboost_available", lambda: False)
    assert isinstance(build_estimator(LogisticConfig()), Pipeline)
    assert isinstance(build_estimator(RandomForestConfig(n_estimators=2)), Pipeline)


def test_available_xgboost_builds_one_vs_rest_without_a_scaler(monkeypatch) -> None:
    monkeypatch.setattr(models, "_xgboost_classifier", lambda: _RecordingBooster)
    pipeline = build_estimator(XGBoostConfig(n_estimators=9, max_depth=2))
    assert [name for name, _ in pipeline.steps] == ["classifier"]
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, OneVsRestClassifier)
    assert isinstance(classifier.estimator, _RecordingBooster)
    assert classifier.estimator.get_params()["n_estimators"] == 9
    assert classifier.estimator.get_params()["max_depth"] == 2
    assert classifier.estimator.get_params()["eval_metric"] == "logloss"


def test_xgboost_describe_reports_the_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(models, "xgboost_available", lambda: False)
    payload = describe_estimator(XGBoostConfig(n_estimators=4))
    assert payload["model_id"] == "xgboost_ovr"
    assert payload["dependency"] == "xgboost"
    assert payload["dependency_available"] is False
    assert payload["scaler"] is None
    assert payload["n_estimators"] == 4


@pytest.mark.parametrize(
    "override",
    [
        {"n_estimators": 0},
        {"max_depth": 0},
        {"learning_rate": 0.0},
        {"subsample": 1.5},
        {"colsample_bytree": -0.1},
        {"reg_lambda": -1.0},
        {"n_jobs": 0},
    ],
)
def test_xgboost_config_rejects_invalid_values(override: dict) -> None:
    with pytest.raises(ValueError):
        XGBoostConfig(**override)


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


def test_every_family_shares_the_build_and_describe_interface() -> None:
    assert SUPPORTED_MODEL_CONFIGS == (LogisticConfig, RandomForestConfig, XGBoostConfig)
    for config in (LogisticConfig(), RandomForestConfig(n_estimators=2), XGBoostConfig()):
        assert isinstance(config.model_id, str) and config.model_id
    assert LogisticConfig().model_id == "logistic_ovr"


def test_unknown_configuration_objects_are_rejected() -> None:
    with pytest.raises(TypeError):
        build_estimator("random forest")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        describe_estimator(object())  # type: ignore[arg-type]


def test_logistic_pipeline_and_defaults_are_unchanged() -> None:
    pipeline = build_estimator(LogisticConfig())
    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, OneVsRestClassifier)
    assert classifier.n_jobs == 1
    logistic = classifier.estimator
    assert isinstance(logistic, LogisticRegression)
    assert logistic.C == 1.0
    assert logistic.class_weight is None
    assert logistic.solver == "liblinear"
    assert logistic.max_iter == 1000
    assert logistic.random_state == 42
    assert describe_estimator()["model_id"] == "logistic_ovr"
