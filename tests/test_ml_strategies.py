"""Tests for the strategy comparison layer (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.feature_selection import (
    ForwardSelectionConfig,
    run_backward_elimination,
    run_forward_selection,
)
from protein_database.ml.splits import make_group_split_plan
from protein_database.ml.strategies import (
    STRATEGY_ALL,
    STRATEGY_BACKWARD,
    STRATEGY_FORWARD,
    STRATEGY_VARIANCE,
    all_features_outcome,
    backward_outcome,
    compare_outcomes,
    feature_stability,
    forward_outcome,
    variance_prefilter_outcome,
)

GROUPS = np.repeat(np.arange(12), 6)
LABELS = ("GO:0003824", "GO:0005488")


def _dataset(seed: int = 3, constant_column: bool = False) -> ProteinDataset:
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 2))
    columns = [signal, noise[:, 0], noise[:, 1]]
    names = ["f_a", "n1", "n2"]
    if constant_column:
        columns.append(np.ones(n))
        names.append("constant")
    return ProteinDataset(
        keys=tuple(
            ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
            for index in range(n)
        ),
        X=np.column_stack(columns),
        y=np.column_stack([signal > 0, signal > 0.5]),
        feature_names=tuple(names),
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )


def _plan(seed: int = 3):
    return make_group_split_plan(GROUPS, seed=seed)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_all_features_baseline_keeps_every_column_in_order() -> None:
    dataset = _dataset()
    outcome = all_features_outcome(dataset, _plan())
    assert outcome.strategy == STRATEGY_ALL
    assert outcome.selected == dataset.feature_names
    assert outcome.n_features == 3
    assert outcome.n_evaluations == 0
    assert outcome.stopped_reason == "not_applicable"


def test_all_features_baseline_includes_constant_columns() -> None:
    dataset = _dataset(constant_column=True)
    outcome = all_features_outcome(dataset, _plan())
    assert "constant" in outcome.selected


def test_variance_baseline_drops_constant_columns() -> None:
    dataset = _dataset(constant_column=True)
    outcome = variance_prefilter_outcome(dataset, _plan())
    assert outcome.strategy == STRATEGY_VARIANCE
    assert "constant" not in outcome.selected
    assert outcome.detail["dropped_by_variance_filter"] == ["constant"]
    assert outcome.n_features == 3


def test_variance_baseline_raises_when_everything_is_constant() -> None:
    dataset = _dataset()
    constant = ProteinDataset(
        keys=dataset.keys,
        X=np.ones_like(dataset.X),
        y=dataset.y,
        feature_names=dataset.feature_names,
        label_names=dataset.label_names,
        report=dataset.report,
    )
    with pytest.raises(ValueError, match="removed every feature"):
        variance_prefilter_outcome(constant, _plan())


# ---------------------------------------------------------------------------
# Strategy outcomes
# ---------------------------------------------------------------------------


def test_forward_and_backward_outcomes_share_the_same_shape() -> None:
    dataset = _dataset()
    plan = _plan()
    config = ForwardSelectionConfig(max_features=2)

    forward = forward_outcome(
        run_forward_selection(dataset, plan, config=config), dataset, plan
    )
    backward = backward_outcome(
        run_backward_elimination(dataset, plan, config=config), dataset, plan
    )

    for outcome in (forward, backward):
        assert outcome.n_features == len(outcome.selected)
        assert outcome.mean_macro_f1 == pytest.approx(
            outcome.evaluation.mean_macro_f1
        )
        assert outcome.std_macro_f1 >= 0.0
        assert outcome.n_evaluations > 0
        json.dumps(outcome.as_dict())


def test_comparison_reports_every_strategy_with_its_own_size() -> None:
    dataset = _dataset(constant_column=True)
    plan = _plan()
    config = ForwardSelectionConfig(max_features=2)

    outcomes = [
        all_features_outcome(dataset, plan, config=config),
        variance_prefilter_outcome(dataset, plan, config=config),
        forward_outcome(
            run_forward_selection(dataset, plan, config=config), dataset, plan
        ),
        backward_outcome(
            run_backward_elimination(dataset, plan, config=config), dataset, plan
        ),
    ]
    report = compare_outcomes(outcomes)

    assert [row["strategy"] for row in report.rows] == [
        STRATEGY_ALL,
        STRATEGY_VARIANCE,
        STRATEGY_FORWARD,
        STRATEGY_BACKWARD,
    ]
    sizes = [row["n_features"] for row in report.rows]
    assert sizes[0] == 4  # includes the constant column
    assert sizes[1] == 3  # constant removed
    assert all(size >= 1 for size in sizes)
    assert all("mean_macro_f1" in row for row in report.rows)
    json.dumps(report.as_dict())


def test_per_fold_stability_is_reported_as_unavailable() -> None:
    dataset = _dataset()
    plan = _plan()
    report = compare_outcomes([all_features_outcome(dataset, plan)])
    assert report.stability["per_fold"]["available"] is False
    assert "not one selection per fold" in report.stability["per_fold"]["reason"]


def test_across_strategy_frequency_uses_the_selected_subsets() -> None:
    dataset = _dataset()
    plan = _plan()
    config = ForwardSelectionConfig(max_features=2)
    outcomes = [
        all_features_outcome(dataset, plan, config=config),
        variance_prefilter_outcome(dataset, plan, config=config),
    ]
    report = compare_outcomes(outcomes)
    frequency = report.stability["across_strategies"]
    assert frequency["n_selections"] == 2
    assert frequency["frequency"]["f_a"]["count"] == 2
    assert frequency["frequency"]["f_a"]["fraction"] == pytest.approx(1.0)


def test_feature_stability_counts_frequencies_explicitly() -> None:
    payload = feature_stability(
        {"fold_0": ("f_a", "f_b"), "fold_1": ("f_a",), "fold_2": ("f_a", "f_c")}
    )
    assert payload["n_selections"] == 3
    assert payload["frequency"]["f_a"] == {"count": 3, "fraction": 1.0}
    assert payload["frequency"]["f_b"] == {"count": 1, "fraction": pytest.approx(1 / 3)}
    assert payload["frequency"]["f_c"]["count"] == 1


def test_compare_rejects_empty_and_duplicate_strategies() -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_outcomes([])

    dataset = _dataset()
    plan = _plan()
    outcome = all_features_outcome(dataset, plan)
    with pytest.raises(ValueError, match="Duplicate strategy"):
        compare_outcomes([outcome, outcome])


def test_all_outcomes_use_the_same_folds() -> None:
    dataset = _dataset()
    plan = _plan()
    config = ForwardSelectionConfig(max_features=2)
    outcomes = [
        all_features_outcome(dataset, plan, config=config),
        variance_prefilter_outcome(dataset, plan, config=config),
        forward_outcome(
            run_forward_selection(dataset, plan, config=config), dataset, plan
        ),
    ]
    fold_names = {
        tuple(fold.fold for fold in outcome.evaluation.fold_scores)
        for outcome in outcomes
    }
    assert len(fold_names) == 1
    assert fold_names.pop() == tuple(fold.name for fold in plan.folds)


def test_outcomes_ignore_the_external_test_block() -> None:
    plan = _plan()
    dataset = _dataset()
    clean = all_features_outcome(dataset, plan)

    tampered = ProteinDataset(
        keys=dataset.keys,
        X=dataset.X.copy(),
        y=dataset.y.copy(),
        feature_names=dataset.feature_names,
        label_names=dataset.label_names,
        report=dataset.report,
    )
    tampered.X[plan.test, :] = 1e6
    tampered.y[plan.test, :] = True
    dirty = all_features_outcome(tampered, plan)

    assert clean.mean_macro_f1 == pytest.approx(dirty.mean_macro_f1)
    assert [fold.macro_f1 for fold in clean.evaluation.fold_scores] == [
        fold.macro_f1 for fold in dirty.evaluation.fold_scores
    ]
