"""Tests for the execution plan (enumeration only, no training)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml import feature_selection as selection_module
from protein_database.ml import models as models_module
from protein_database.ml.experiment_config import SelectionSpec, model_spec
from protein_database.ml.experiment_guard import forbid_every_row
from protein_database.ml.experiment_plan import (
    STATUS_PLANNED,
    STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
    SUBSET_ALL,
    SUBSET_SELECTION,
    SUBSET_VARIANCE,
    SubsetDefinition,
    build_execution_plan,
    estimate_evaluation_cost,
    estimate_selection_cost,
)
from protein_database.ml.feature_selection import ForwardSelectionConfig
from protein_database.ml.models import LogisticConfig, RandomForestConfig, XGBoostConfig
from protein_database.ml.splits import Fold, SplitPlan
from tests.experiment_helpers import (
    LABELS,
    make_dataset,
    make_plan,
    plan_for,
    small_config,
    tiny_config,
)


class _NoTestAccess:
    """A SplitPlan proxy that fails loudly if ``plan.test`` is ever read."""

    def __init__(self, plan: SplitPlan) -> None:
        self._plan = plan
        self.accesses = 0

    @property
    def test(self):
        self.accesses += 1
        raise AssertionError("the planner/report read plan.test")

    @property
    def folds(self):
        return self._plan.folds

    @property
    def n_folds(self) -> int:
        return self._plan.n_folds

    @property
    def seed(self) -> int:
        return self._plan.seed

    @property
    def test_size(self) -> float:
        return self._plan.test_size

    @property
    def n_groups_test(self) -> int:
        return self._plan.n_groups_test

    @property
    def n_groups_dev(self) -> int:
        return self._plan.n_groups_dev


# ---------------------------------------------------------------------------
# Determinism and ordering
# ---------------------------------------------------------------------------


def test_plan_is_deterministic_for_the_same_inputs() -> None:
    config, dataset, plan = small_config(), make_dataset(), make_plan()
    first = build_execution_plan(config, dataset, plan)
    second = build_execution_plan(config, dataset, plan)
    assert first.plan_id == second.plan_id
    assert first.task_ids == second.task_ids
    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(
        second.as_dict(), sort_keys=True
    )


def test_plan_id_changes_with_the_configuration_and_the_folds() -> None:
    dataset, plan = make_dataset(), make_plan()
    base = build_execution_plan(small_config(), dataset, plan)
    other_seed = build_execution_plan(small_config(seed=7), dataset, plan)
    other_folds = build_execution_plan(small_config(), dataset, make_plan(seed=11))
    assert base.plan_id != other_seed.plan_id
    assert base.plan_id != other_folds.plan_id
    assert base.folds_id != other_folds.folds_id


def test_task_order_is_selection_first_then_declared_strategy_order() -> None:
    tasks = plan_for(small_config()).tasks
    assert [task.kind for task in tasks[:2]] == ["selection", "selection"]
    assert [task.strategy for task in tasks[:2]] == ["forward", "backward"]
    evaluation = [task for task in tasks if task.kind == "evaluation"]
    assert [(task.strategy, task.model_label) for task in evaluation[:2]] == [
        ("all_features", "logit"),
        ("all_features", "forest"),
    ]
    assert [task.order for task in tasks] == list(range(len(tasks)))


def test_selection_evaluations_depend_on_their_selection_task() -> None:
    plan = plan_for(small_config())
    selection = {
        task.strategy: task.task_id for task in plan.tasks if task.kind == "selection"
    }
    for task in plan.tasks:
        if task.kind != "evaluation":
            continue
        if task.strategy in ("forward", "backward"):
            assert task.requires == (selection[task.strategy],)
            assert task.subset.kind == SUBSET_SELECTION
        else:
            assert task.requires == ()


def test_plain_strategies_have_no_selection_dependency() -> None:
    plan = plan_for(tiny_config())
    (task,) = plan.tasks
    assert task.subset.kind == SUBSET_ALL
    assert task.requires == ()
    assert task.cost.fold_evaluations == plan.n_folds


# ---------------------------------------------------------------------------
# Planning does not train and does not see the test block
# ---------------------------------------------------------------------------


def test_planning_never_trains(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("planning must not fit or score anything")

    monkeypatch.setattr(selection_module, "evaluate_feature_subset", explode)
    monkeypatch.setattr(selection_module, "run_forward_selection", explode)
    monkeypatch.setattr(selection_module, "run_backward_elimination", explode)
    monkeypatch.setattr(models_module, "build_estimator", explode)

    plan = plan_for(small_config())
    assert len(plan.tasks) == 10
    assert plan.total_cost().estimator_fits > 0


def test_planning_does_not_read_the_data_matrix() -> None:
    config, plan = small_config(), make_plan()
    clean = build_execution_plan(config, make_dataset(), plan)
    blind = build_execution_plan(config, forbid_every_row(make_dataset()), plan)
    assert blind.plan_id == clean.plan_id


def test_planning_never_touches_plan_test() -> None:
    proxy = _NoTestAccess(make_plan())
    plan = build_execution_plan(tiny_config(), make_dataset(), proxy)
    assert plan.n_samples == 72
    assert proxy.accesses == 0


# ---------------------------------------------------------------------------
# Availability of optional dependencies
# ---------------------------------------------------------------------------


def test_xgboost_is_planned_as_skipped_when_the_dependency_is_absent() -> None:
    config = default_experiment_config_for_xgboost()
    plan = build_execution_plan(
        config, make_dataset(), make_plan(), dependency_available=lambda module: False
    )
    skipped = plan.tasks_with_status(STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY)
    assert len(skipped) == len(config.strategies)
    assert all("xgboost" in task.skip_reason for task in skipped)
    assert all(task.model_label == "xgboost_base" for task in skipped)
    assert all(task.status != STATUS_PLANNED for task in skipped)


def test_available_dependency_leaves_the_tasks_planned() -> None:
    config = default_experiment_config_for_xgboost()
    plan = build_execution_plan(
        config, make_dataset(), make_plan(), dependency_available=lambda module: True
    )
    assert plan.tasks_with_status(STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY) == ()
    assert len(plan.tasks) == len(config.selection) + len(config.strategies)


def test_default_probe_reflects_the_real_environment() -> None:
    import importlib.util

    from protein_database.ml.experiment_plan import default_dependency_probe

    expected = importlib.util.find_spec("xgboost") is not None
    assert default_dependency_probe("xgboost") is expected


def default_experiment_config_for_xgboost():
    return small_config(
        models=(model_spec("xgboost_base", XGBoostConfig()),),
    )


# ---------------------------------------------------------------------------
# Cost estimates
# ---------------------------------------------------------------------------


def test_evaluation_cost_matches_the_folds_and_labels() -> None:
    cost = estimate_evaluation_cost(n_folds=3, n_labels=3)
    assert cost.tasks == 1
    assert cost.fold_evaluations == 3
    assert cost.estimator_fits == 3
    assert cost.binary_fits == 9
    assert cost.selection_candidate_evaluations == 0
    assert cost.exact is False
    assert any("upper bound" in note for note in cost.notes)


def test_forward_selection_cost_is_an_upper_bound() -> None:
    spec = SelectionSpec(
        strategy="forward", config=ForwardSelectionConfig(max_features=2)
    )
    cost = estimate_selection_cost(spec, n_features=4, n_folds=3, n_labels=3)
    # round 1 evaluates 4 candidates, round 2 evaluates 3 -> 7 candidates x 3 folds
    assert cost.selection_candidate_evaluations == 7
    assert cost.fold_evaluations == 21
    assert cost.estimator_fits == 21
    assert cost.binary_fits == 63
    assert cost.exact is False


def test_backward_selection_cost_counts_the_reference_evaluation() -> None:
    spec = SelectionSpec(
        strategy="backward",
        config=ForwardSelectionConfig(max_features=2),
        min_features=2,
    )
    cost = estimate_selection_cost(spec, n_features=4, n_folds=3, n_labels=3)
    assert cost.selection_candidate_evaluations == 8
    assert cost.fold_evaluations == 24
    assert cost.estimator_fits == 27  # 24 candidate fits + 3 reference fits


def test_max_evaluations_caps_the_selection_estimate() -> None:
    spec = SelectionSpec(
        strategy="forward",
        config=ForwardSelectionConfig(max_features=10, max_evaluations=3),
    )
    cost = estimate_selection_cost(spec, n_features=4, n_folds=3, n_labels=3)
    assert cost.selection_candidate_evaluations == 3
    assert cost.fold_evaluations == 9


def test_cost_estimators_reject_impossible_dimensions() -> None:
    with pytest.raises(ValueError):
        estimate_selection_cost(
            SelectionSpec(strategy="forward"), n_features=0, n_folds=3, n_labels=1
        )
    with pytest.raises(ValueError):
        estimate_evaluation_cost(n_folds=0, n_labels=1)




# ---------------------------------------------------------------------------
# Validation and serialisation
# ---------------------------------------------------------------------------


def _plan_with(folds: tuple[Fold, ...]) -> SplitPlan:
    return SplitPlan(
        test=np.array([0, 1]),
        folds=folds,
        seed=42,
        test_size=0.2,
        n_groups_test=1,
        n_groups_dev=1,
    )


def test_out_of_range_fold_indices_are_rejected() -> None:
    broken = _plan_with(
        (Fold(index=0, train=np.array([0, 1]), validation=np.array([999])),)
    )
    with pytest.raises(ValueError, match="outside"):
        build_execution_plan(tiny_config(), make_dataset(), broken)


def test_overlapping_train_and_validation_are_rejected() -> None:
    broken = _plan_with(
        (Fold(index=0, train=np.array([0, 1]), validation=np.array([1, 2])),)
    )
    with pytest.raises(ValueError, match="shares indices"):
        build_execution_plan(tiny_config(), make_dataset(), broken)


def test_empty_fold_blocks_are_rejected() -> None:
    broken = _plan_with(
        (Fold(index=0, train=np.array([], dtype=int), validation=np.array([1])),)
    )
    with pytest.raises(ValueError, match="empty"):
        build_execution_plan(tiny_config(), make_dataset(), broken)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "nowhere"},
        {"kind": SUBSET_SELECTION},
        {"kind": SUBSET_ALL, "selection_id": "forward/abc"},
        {"kind": SUBSET_VARIANCE, "selection_id": "forward/abc"},
    ],
)
def test_subset_definitions_are_validated(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SubsetDefinition(**kwargs)


def test_plan_serialises_with_consistent_counts() -> None:
    plan = plan_for(small_config())
    payload = plan.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert sum(payload["status_counts"].values()) == payload["n_tasks"] == 10
    assert payload["estimated_cost"]["tasks"] == 10
    assert any("test block" in note for note in payload["notes"])
    assert payload["n_labels"] == len(LABELS)
    assert payload["n_folds"] == 3


def test_skipped_tasks_are_excluded_from_the_executable_cost() -> None:
    config = default_experiment_config_for_xgboost()
    plan = build_execution_plan(
        config, make_dataset(), make_plan(), dependency_available=lambda module: False
    )
    # only the two selection tasks remain executable; every evaluation is skipped
    assert plan.total_cost().tasks == len(config.selection)
    assert plan.total_cost(include_skipped=True).tasks == (
        len(config.selection) + len(config.strategies)
    )


def test_unknown_task_ids_are_reported() -> None:
    plan = plan_for(tiny_config())
    with pytest.raises(KeyError):
        plan.get("does-not-exist")
    assert plan.get(plan.task_ids[0]).task_id == plan.task_ids[0]
