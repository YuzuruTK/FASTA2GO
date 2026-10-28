"""Tests for backward elimination (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.feature_selection import (
    CANDIDATE_COLUMNS,
    STOP_BUDGET,
    STOP_MIN_FEATURES,
    STOP_MIN_IMPROVEMENT,
    STOP_NO_EVALUABLE_LABELS,
    ForwardSelectionConfig,
    run_backward_elimination,
)
from protein_database.ml.splits import Fold, SplitPlan, make_group_split_plan

GROUPS = np.repeat(np.arange(12), 6)
LABELS = ("GO:0003824", "GO:0005488")


def _dataset(seed: int, n_noise: int = 3) -> ProteinDataset:
    """One informative feature plus ``n_noise`` noise columns; labels follow it."""
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    columns = [rng.normal(size=n) for _ in range(1 + n_noise)]
    names = ("f_a",) + tuple(f"n{index}" for index in range(1, n_noise + 1))
    keys = tuple(
        ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
        for index in range(n)
    )
    return ProteinDataset(
        keys=keys,
        X=np.column_stack(columns),
        y=np.column_stack([columns[0] > 0, columns[0] > 0.5]),
        feature_names=names,
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )


def _plan(seed: int = 3) -> SplitPlan:
    return make_group_split_plan(GROUPS, seed=seed)


def test_removes_uninformative_features_first() -> None:
    result = run_backward_elimination(_dataset(3), _plan())
    # the two noise columns whose removal improved the score go first
    assert result.removed == ("n2", "n3")
    assert result.remaining == ("f_a", "n1")
    assert result.stopped_reason == STOP_MIN_IMPROVEMENT


def test_rejected_round_is_recorded_and_stops_the_run() -> None:
    result = run_backward_elimination(_dataset(3), _plan())
    last = result.history[-1]
    assert last.accepted is False
    assert last.gain is not None and last.gain < 0
    assert len(result.history) == 3
    assert result.history[-1].round == 3


def test_no_round_after_the_stop() -> None:
    history = run_backward_elimination(_dataset(3), _plan()).history
    rejected = [index for index, round_ in enumerate(history) if not round_.accepted]
    assert rejected == [len(history) - 1]


def test_stops_immediately_when_no_removal_improves() -> None:
    """Three identical columns: every removal leaves the score unchanged."""
    rng = np.random.default_rng(2)
    n = GROUPS.size
    signal = rng.normal(size=n)
    dataset = ProteinDataset(
        keys=tuple(
            ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
            for index in range(n)
        ),
        X=np.column_stack([signal, signal.copy(), signal.copy()]),
        y=np.column_stack([signal > 0, signal > 0.5]),
        feature_names=("a", "b", "c"),
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )
    result = run_backward_elimination(dataset, _plan())

    assert len(result.history) == 1
    assert result.history[0].accepted is False
    assert result.history[0].gain == pytest.approx(0.0)
    assert result.history[0].tied_candidates == ("a", "b", "c")
    assert result.removed == ()
    assert result.remaining == ("a", "b", "c")
    assert result.stopped_reason == STOP_MIN_IMPROVEMENT


def test_min_features_floor_stops_the_run() -> None:
    result = run_backward_elimination(_dataset(1), _plan(), min_features=1)
    assert result.stopped_reason == STOP_MIN_FEATURES
    assert len(result.remaining) == 1
    assert result.remaining == ("f_a",)


def test_larger_min_features_stops_earlier() -> None:
    result = run_backward_elimination(_dataset(1), _plan(), min_features=2)
    assert len(result.remaining) >= 2
    assert len(result.remaining) + len(result.removed) == 4


def test_ties_are_broken_by_reference_order() -> None:
    rng = np.random.default_rng(3)
    n = GROUPS.size
    signal = rng.normal(size=n)
    duplicate = rng.normal(size=n)
    dataset = ProteinDataset(
        keys=tuple(
            ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
            for index in range(n)
        ),
        X=np.column_stack([signal, duplicate, duplicate.copy()]),
        y=np.column_stack([signal > 0, signal > 0.5]),
        feature_names=("f_a", "dup_first", "dup_second"),
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )
    result = run_backward_elimination(
        dataset, _plan(), config=ForwardSelectionConfig(min_improvement=0.0)
    )
    first = result.history[0]
    assert first.tied_candidates == ("dup_first", "dup_second")
    assert first.removed == "dup_first"


def test_budget_stops_before_a_partial_round() -> None:
    result = run_backward_elimination(
        _dataset(3), _plan(), config=ForwardSelectionConfig(max_evaluations=2)
    )
    assert result.stopped_reason == STOP_BUDGET
    assert result.history == ()
    assert result.remaining == ("f_a", "n1", "n2", "n3")


def test_result_is_deterministic() -> None:
    first = run_backward_elimination(_dataset(3), _plan())
    second = run_backward_elimination(_dataset(3), _plan())
    assert first.remaining == second.remaining
    assert first.removed == second.removed
    assert [(round_.removed, round_.score) for round_ in first.history] == [
        (round_.removed, round_.score) for round_ in second.history
    ]


def test_remaining_features_keep_the_reference_order() -> None:
    dataset = _dataset(3)
    result = run_backward_elimination(dataset, _plan())
    order = [name for name in dataset.feature_names if name in set(result.remaining)]
    assert list(result.remaining) == order


def test_history_records_candidates_folds_and_spread() -> None:
    plan = _plan()
    result = run_backward_elimination(_dataset(3), plan)
    first = result.history[0]
    assert first.n_candidates_evaluated == len(first.candidates) == 4
    assert [fold.fold for fold in first.fold_scores] == [
        fold.name for fold in plan.folds
    ]
    assert first.score_std >= 0.0
    assert first.reference_score is not None
    assert first.score == pytest.approx(
        float(np.mean([fold.macro_f1 for fold in first.fold_scores]))
    )


def test_candidate_rows_follow_a_stable_schema() -> None:
    rows = run_backward_elimination(_dataset(3), _plan()).candidate_rows()
    assert rows
    assert [list(row) for row in rows][0] == list(CANDIDATE_COLUMNS)


def test_report_is_json_serialisable() -> None:
    payload = run_backward_elimination(_dataset(3), _plan()).report()
    assert payload["stopped_reason"] == STOP_MIN_IMPROVEMENT
    assert payload["removed"] == ["n2", "n3"]
    json.dumps(payload)


def test_rejects_invalid_min_features() -> None:
    with pytest.raises(ValueError, match="min_features"):
        run_backward_elimination(_dataset(3), _plan(), min_features=0)


def test_rejects_plan_indices_outside_the_dataset() -> None:
    dataset = _dataset(3)
    broken = SplitPlan(
        test=np.array([0, 1]),
        folds=(Fold(0, np.array([2, 3]), np.array([dataset.n_samples + 5])),),
        seed=0,
        test_size=0.2,
        n_groups_test=1,
        n_groups_dev=1,
    )
    with pytest.raises(ValueError, match="outside"):
        run_backward_elimination(dataset, broken)


def test_no_evaluable_labels() -> None:
    dataset = _dataset(3)
    empty = ProteinDataset(
        keys=dataset.keys,
        X=dataset.X,
        y=np.zeros_like(dataset.y),
        feature_names=dataset.feature_names,
        label_names=dataset.label_names,
        report=dataset.report,
    )
    result = run_backward_elimination(empty, _plan())
    assert result.stopped_reason == STOP_NO_EVALUABLE_LABELS
    assert result.remaining == ()
    assert result.excluded_folds == ("fold_0", "fold_1", "fold_2")
