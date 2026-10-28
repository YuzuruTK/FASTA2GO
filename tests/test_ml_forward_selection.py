"""Tests for forward selection over grouped inner folds (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml.dataset import (
    DatasetError,
    DatasetReport,
    ProteinDataset,
    ProteinKey,
)
from protein_database.ml.feature_selection import (
    CANDIDATE_COLUMNS,
    STOP_BUDGET,
    STOP_MAX_FEATURES,
    STOP_MIN_IMPROVEMENT,
    STOP_NO_CANDIDATES,
    STOP_NO_EVALUABLE_LABELS,
    ForwardSelectionConfig,
    run_forward_selection,
)
from protein_database.ml.splits import Fold, SplitPlan, make_group_split_plan

GROUPS = np.repeat(np.arange(12), 6)
LABELS = ("GO:0003824", "GO:0005488", "GO:0003677")


def _dataset(X, y, names, labels=LABELS[:2], n=None) -> ProteinDataset:
    n = n or X.shape[0]
    keys = tuple(
        ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
        for index in range(n)
    )
    return ProteinDataset(
        keys=keys,
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=bool),
        feature_names=tuple(names),
        label_names=tuple(labels),
        report=DatasetReport(n, (), (), (), {}, 0),
    )


def _plan(seed: int = 3) -> SplitPlan:
    return make_group_split_plan(GROUPS, seed=seed)


def _signal_dataset(seed: int = 7) -> ProteinDataset:
    """One informative feature and three noise columns; both labels follow it."""
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    X = np.column_stack([signal, noise[:, 0], noise[:, 1], noise[:, 2]])
    y = np.column_stack([signal > 0, signal > 0.5])
    return _dataset(X, y, ("f_signal", "n1", "n2", "n3"))


def _two_label_dataset(seed: int = 11) -> ProteinDataset:
    """Each of two features explains one label; a third is noise."""
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    first, second, third = (rng.normal(size=n) for _ in range(3))
    X = np.column_stack([first, second, third])
    y = np.column_stack([first > 0, second > 0])
    return _dataset(X, y, ("f_a", "f_b", "f_c"))


# ---------------------------------------------------------------------------
# Selection behaviour
# ---------------------------------------------------------------------------


def test_selects_the_informative_feature_first() -> None:
    result = run_forward_selection(_signal_dataset(), _plan())
    assert result.selection.selected[0] == "f_signal"


def test_stops_when_no_candidate_improves() -> None:
    result = run_forward_selection(_signal_dataset(), _plan())
    assert result.selection.stopped_reason == STOP_MIN_IMPROVEMENT
    assert len(result.selection.selected) == 1
    assert result.selection.history[-1].accepted is False


def test_no_round_is_recorded_after_the_stop() -> None:
    history = run_forward_selection(_signal_dataset(), _plan()).selection.history
    rejected = [index for index, round_ in enumerate(history) if not round_.accepted]
    assert len(rejected) == 1
    assert rejected[0] == len(history) - 1


def test_ties_are_broken_by_reference_order() -> None:
    rng = np.random.default_rng(5)
    signal = rng.normal(size=GROUPS.size)
    dataset = _dataset(
        np.column_stack([signal, signal.copy()]),
        np.column_stack([signal > 0, signal > 0.5]),
        ("f_first", "f_second"),
    )
    result = run_forward_selection(dataset, _plan())

    first_round = result.selection.history[0]
    assert first_round.tied_candidates == ("f_first", "f_second")
    assert first_round.candidate == "f_first"
    scores = {candidate.feature: candidate.score for candidate in first_round.candidates}
    assert scores["f_first"] == pytest.approx(scores["f_second"])
    assert result.selection.selected[0] == "f_first"


def test_respects_the_max_features_ceiling() -> None:
    result = run_forward_selection(
        _two_label_dataset(),
        _plan(),
        config=ForwardSelectionConfig(max_features=2, min_improvement=0.0),
    )
    assert result.selection.selected == ("f_a", "f_b")
    assert result.selection.stopped_reason == STOP_MAX_FEATURES


def test_stops_when_candidates_run_out() -> None:
    plan = _plan()
    signal = np.random.default_rng(7).normal(size=GROUPS.size)
    # constant in fold 0's training rows, so it never becomes a candidate
    blocker = np.zeros(GROUPS.size)
    blocker[plan.folds[0].validation] = 1.0
    dataset = _dataset(
        np.column_stack([signal, blocker]),
        np.column_stack([signal > 0, signal > 0.5]),
        ("f_a", "f_out"),
    )
    result = run_forward_selection(
        dataset, plan, config=ForwardSelectionConfig(max_features=5)
    )
    assert result.prefilter_dropped == ("f_out",)
    assert result.selection.stopped_reason == STOP_NO_CANDIDATES


def test_all_constant_features_stop_without_rounds() -> None:
    signal = np.random.default_rng(7).normal(size=GROUPS.size)
    dataset = _dataset(
        np.ones((GROUPS.size, 2)),
        np.column_stack([signal > 0, signal > 0.5]),
        ("k1", "k2"),
    )
    result = run_forward_selection(dataset, _plan())
    assert result.selection.selected == ()
    assert result.selection.history == ()
    assert result.selection.stopped_reason == STOP_NO_CANDIDATES
    assert result.prefilter_dropped == ("k1", "k2")


def test_budget_stops_before_a_partial_round() -> None:
    result = run_forward_selection(
        _signal_dataset(),
        _plan(),
        config=ForwardSelectionConfig(max_evaluations=2),
    )
    assert result.selection.stopped_reason == STOP_BUDGET
    assert result.selection.history == ()
    assert result.selection.selected == ()


def test_no_evaluable_labels_anywhere() -> None:
    dataset = _dataset(
        np.random.default_rng(0).normal(size=(GROUPS.size, 2)),
        np.zeros((GROUPS.size, 2), dtype=bool),
        ("a", "b"),
    )
    result = run_forward_selection(dataset, _plan())
    assert result.selection.stopped_reason == STOP_NO_EVALUABLE_LABELS
    assert result.selection.selected == ()
    assert result.excluded_folds == ("fold_0", "fold_1", "fold_2")


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_history_records_candidates_folds_and_spread() -> None:
    plan = _plan()
    result = run_forward_selection(_signal_dataset(), plan)
    first = result.selection.history[0]

    assert first.n_candidates_evaluated == len(first.candidates) == 4
    assert len(first.fold_scores) == plan.n_folds
    assert [fold.fold for fold in first.fold_scores] == [
        fold.name for fold in plan.folds
    ]
    assert first.score_std >= 0.0
    assert first.score == pytest.approx(
        float(np.mean([fold.macro_f1 for fold in first.fold_scores]))
    )
    assert first.n_model_fits == sum(
        fold.n_model_fits for fold in first.fold_scores
    )


def test_candidate_rows_follow_a_stable_schema() -> None:
    result = run_forward_selection(_signal_dataset(), _plan())
    rows = result.selection.candidate_rows()
    assert rows
    assert [list(row) for row in rows][0] == list(CANDIDATE_COLUMNS)
    assert {row["round"] for row in rows} == {
        round_.round for round_ in result.selection.history
    }


def test_the_same_folds_are_used_for_every_candidate() -> None:
    result = run_forward_selection(_signal_dataset(), _plan())
    names = {
        tuple(fold.fold for fold in round_.fold_scores)
        for round_ in result.selection.history
    }
    assert len(names) == 1


def test_labels_without_support_are_excluded_and_recorded() -> None:
    plan = _plan()
    rng = np.random.default_rng(7)
    signal = rng.normal(size=GROUPS.size)
    y = np.zeros((GROUPS.size, 3), dtype=bool)
    y[:, 0] = signal > 0
    y[:, 1] = signal > 0.5
    # third label only exists in one fold's validation rows
    y[plan.folds[0].validation, 2] = True

    result = run_forward_selection(
        _dataset(np.column_stack([signal, rng.normal(size=GROUPS.size)]), y,
                 ("f_signal", "f_noise"), labels=LABELS),
        plan,
    )
    for fold in result.selection.history[0].fold_scores:
        assert fold.n_labels_evaluated == 2
        assert fold.n_labels_total == 3
        assert LABELS[2] in fold.excluded_labels


def test_test_block_is_never_read() -> None:
    plan = _plan()
    dataset = _signal_dataset()
    clean = run_forward_selection(dataset, plan)

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
    dirty = run_forward_selection(tampered, plan)

    assert clean.selection.selected == dirty.selection.selected
    assert [
        (round_.candidate, round_.score) for round_ in clean.selection.history
    ] == [(round_.candidate, round_.score) for round_ in dirty.selection.history]


def test_result_is_deterministic() -> None:
    first = run_forward_selection(_signal_dataset(), _plan())
    second = run_forward_selection(_signal_dataset(), _plan())
    assert first.selection.selected == second.selection.selected
    assert [
        (round_.candidate, round_.score, round_.accepted)
        for round_ in first.selection.history
    ] == [
        (round_.candidate, round_.score, round_.accepted)
        for round_ in second.selection.history
    ]


def test_report_is_json_serialisable() -> None:
    payload = run_forward_selection(_signal_dataset(), _plan()).report()
    assert payload["selected"]
    assert payload["stopped_reason"] == STOP_MIN_IMPROVEMENT
    assert payload["config"]["estimator"]["type"] == (
        "OneVsRestClassifier(LogisticRegression)"
    )
    json.dumps(payload)


def test_feature_names_and_order_are_preserved() -> None:
    dataset = _signal_dataset()
    result = run_forward_selection(dataset, _plan())
    assert result.feature_names == dataset.feature_names
    assert result.prefilter_kept == dataset.feature_names


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_finite_features() -> None:
    dataset = _signal_dataset()
    broken = ProteinDataset(
        keys=dataset.keys,
        X=np.where(np.arange(dataset.n_features) == 1, np.inf, dataset.X),
        y=dataset.y,
        feature_names=dataset.feature_names,
        label_names=dataset.label_names,
        report=dataset.report,
    )
    with pytest.raises(DatasetError, match="non-finite"):
        run_forward_selection(broken, _plan())


def test_rejects_plan_indices_outside_the_dataset() -> None:
    dataset = _signal_dataset()
    plan = _plan()
    broken = SplitPlan(
        test=plan.test,
        folds=(
            Fold(0, np.array([0, 1, 2]), np.array([dataset.n_samples + 5])),
        ),
        seed=plan.seed,
        test_size=plan.test_size,
        n_groups_test=plan.n_groups_test,
        n_groups_dev=plan.n_groups_dev,
    )
    with pytest.raises(ValueError, match="outside"):
        run_forward_selection(dataset, broken)


@pytest.mark.parametrize(
    "override",
    [
        {"max_features": 0},
        {"max_features": 2.5},
        {"min_improvement": -0.1},
        {"min_improvement": float("nan")},
        {"max_evaluations": 0},
        {"variance_threshold": -1.0},
    ],
)
def test_config_rejects_invalid_values(override: dict) -> None:
    with pytest.raises(ValueError):
        ForwardSelectionConfig(**override)


def test_masks_keep_fitted_labels_separate_from_evaluated_labels() -> None:
    """A label needs support in the training block to be fitted at all."""
    result = run_forward_selection(_signal_dataset(), _plan())
    for fold in result.selection.history[0].fold_scores:
        assert fold.n_model_fits >= fold.n_labels_evaluated
        assert fold.n_fit_samples > 0
        assert fold.n_validation_samples > 0
        assert fold.n_fit_samples + fold.n_validation_samples < GROUPS.size
