"""Regression tests for the scikit-learn behaviours measured in Spike S0.

Each test pins a contract the FASTA2GO pipeline depends on, so that upgrading
scikit-learn fails loudly instead of changing methodology silently:

* ``cv=int`` on a multilabel target resolves to ungrouped ``KFold`` (F1);
* group-aware splitters must be passed as pre-computed index pairs (F4);
* stratified splitters reject multilabel targets (F2);
* ``OneVsRestClassifier`` tolerates a constant label with a warning and a
  constant-false prediction instead of raising (F6).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    check_cv,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _data(
    n_groups: int = 8,
    per_group: int = 6,
    n_features: int = 4,
    n_labels: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered-by-group synthetic data with an learnable multilabel signal."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(groups.size, n_features))
    y = np.column_stack([X[:, k] > 0.0 for k in range(n_labels)])
    return X, y, groups


def _estimator() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=200, solver="liblinear"), n_jobs=1
                ),
            ),
        ]
    )


def _fit_sfs(sfs: SequentialFeatureSelector, X, y) -> SequentialFeatureSelector:
    """Fit while ignoring convergence noise unrelated to what is asserted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sfs.fit(X, y)
    return sfs


def test_check_cv_resolves_to_kfold_for_multilabel() -> None:
    X, y, _ = _data()
    resolved = check_cv(3, y, classifier=True)
    assert type(resolved).__name__ == "KFold"
    assert not isinstance(resolved, StratifiedKFold)
    assert not isinstance(resolved, StratifiedGroupKFold)


def test_check_cv_still_stratifies_binary_targets() -> None:
    """Contrast: only the multilabel target loses stratification."""
    X, y, _ = _data(n_labels=1)
    resolved = check_cv(3, y[:, 0].astype(int), classifier=True)
    assert isinstance(resolved, StratifiedKFold)


def test_integer_cv_leaks_groups_on_grouped_data() -> None:
    """The practical consequence of F1: every fold shares groups."""
    X, y, groups = _data()
    resolved = check_cv(3, y, classifier=True)
    shared = [
        int(np.intersect1d(groups[train], groups[test]).size)
        for train, test in resolved.split(X, y)
    ]
    assert min(shared) > 0


def test_stratified_kfold_rejects_multilabel() -> None:
    X, y, _ = _data()
    with pytest.raises(ValueError, match="Supported target types"):
        list(StratifiedKFold(n_splits=2).split(X, y))


def test_stratified_group_kfold_rejects_multilabel() -> None:
    X, y, groups = _data()
    with pytest.raises(ValueError, match="Supported target types"):
        list(StratifiedGroupKFold(n_splits=2).split(X, y, groups))


def test_sfs_has_no_groups_parameter() -> None:
    """Without a groups parameter, SFS cannot do group-aware CV (F4)."""
    params = SequentialFeatureSelector(_estimator()).get_params()
    assert "groups" not in params


def test_sfs_rejects_group_splitter_instance() -> None:
    X, y, _ = _data()
    with pytest.raises(ValueError, match="groups"):
        _fit_sfs(
            SequentialFeatureSelector(
                _estimator(),
                n_features_to_select=2,
                cv=GroupKFold(n_splits=2),
            ),
            X,
            y,
        )


def test_sfs_accepts_precomputed_group_splits() -> None:
    X, y, groups = _data()
    splits = list(
        GroupShuffleSplit(n_splits=2, test_size=0.25, random_state=0).split(
            X, y, groups
        )
    )
    sfs = _fit_sfs(
        SequentialFeatureSelector(
            _estimator(), n_features_to_select=2, cv=splits
        ),
        X,
        y,
    )
    assert sfs.get_support(indices=True).size == 2


def test_sfs_reports_only_a_mask_without_selection_order() -> None:
    """The audit trail cannot be rebuilt from SFS output (F8)."""
    X, y, groups = _data()
    splits = list(
        GroupShuffleSplit(n_splits=2, test_size=0.25, random_state=0).split(
            X, y, groups
        )
    )
    sfs = _fit_sfs(
        SequentialFeatureSelector(
            _estimator(), n_features_to_select=2, cv=splits
        ),
        X,
        y,
    )
    support = sfs.get_support(indices=True)
    assert np.all(np.diff(support) > 0)  # ascending indices, not selection order
    assert not hasattr(sfs, "history_")
    assert not hasattr(sfs, "n_evaluations_")


def test_ovr_constant_label_warns_and_predicts_false() -> None:
    """F6: a single-class label is silent, not an exception."""
    X, y, _ = _data()
    y = y.copy()
    y[:, 0] = False
    with pytest.warns(UserWarning, match="present in all training examples"):
        model = _estimator().fit(X, y)
    predictions = model.predict(X)
    assert not predictions[:, 0].any()
    assert predictions.shape == y.shape
