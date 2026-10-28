"""Tests for the experiment runner, its states and its reports (synthetic only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml import experiment_runner as runner_module
from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.experiment_budget import RunBudget
from protein_database.ml.experiment_config import FailurePolicy, model_spec
from protein_database.ml.experiment_guard import guard_dataset
from protein_database.ml.experiment_plan import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PLANNED,
    STATUS_SKIPPED_BUDGET,
    STATUS_SKIPPED_DEPENDENCY_FAILURE,
    STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
)
from protein_database.ml.experiment_runner import (
    ExperimentRunner,
    ResumeMismatchError,
    TaskResult,
    TaskStore,
)
from protein_database.ml.feature_selection import evaluate_feature_subset
from protein_database.ml.models import (
    LogisticConfig,
    MissingDependencyError,
    RandomForestConfig,
    XGBoostConfig,
)
from tests.experiment_helpers import (
    FEATURES,
    make_dataset,
    make_plan,
    small_config,
    tiny_config,
)


def _runner(config, *, budget=None, dataset=None, plan=None, **kwargs) -> ExperimentRunner:
    return ExperimentRunner(
        config,
        dataset if dataset is not None else make_dataset(),
        plan if plan is not None else make_plan(),
        budget=budget,
        **kwargs,
    )


def _status(report: dict) -> dict[str, int]:
    return report["status_counts"]


# ---------------------------------------------------------------------------
# Planning, manifest and reporting without execution
# ---------------------------------------------------------------------------


def test_plan_and_manifest_need_no_training(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("planning/manifest must not fit or score anything")

    monkeypatch.setattr(runner_module, "evaluate_feature_subset", explode)
    monkeypatch.setattr(runner_module, "run_forward_selection", explode)
    monkeypatch.setattr(runner_module, "run_backward_elimination", explode)

    runner = _runner(small_config())
    plan = runner.plan()
    manifest = runner.manifest(created_at="2026-01-01T00:00:00+00:00")
    assert plan.plan_id == runner.plan().plan_id
    assert manifest["plan_id"] == plan.plan_id
    assert manifest["protocol_version"] == plan.protocol_version
    assert manifest["split_plan"]["n_folds"] == 3
    assert manifest["environment"]["packages"]["scikit-learn"]
    assert manifest["budget"]["max_tasks"] is None
    assert manifest["catalog"]["n_tasks"] == 10
    assert json.loads(json.dumps(manifest)) == manifest


def test_created_at_is_metadata_and_not_identity() -> None:
    runner = _runner(tiny_config())
    first = runner.manifest(created_at="2026-01-01T00:00:00+00:00")
    second = runner.manifest(created_at="2026-12-31T23:59:59+00:00")
    assert first["plan_id"] == second["plan_id"]
    assert first["config_id"] == second["config_id"]
    assert first["created_at"] != second["created_at"]


def test_report_is_explicit_when_nothing_completed() -> None:
    # one task needs 3 fold evaluations; the budget allows 1, so nothing starts
    report = _runner(tiny_config(), budget=RunBudget(max_fold_evaluations=1)).run(
        write_artifacts=False
    )
    assert report["results_available"] is False
    assert report["rows"] == []
    assert "no results available" in report["results_statement"]
    assert report["external_test"].startswith("not evaluated")
    assert _status(report)["skipped_budget"] == 1
    assert report["tasks"][0]["result"] is None
    assert report["observed_cost"]["consumed"]["tasks_started"] == 0


# ---------------------------------------------------------------------------
# Execution on synthetic data
# ---------------------------------------------------------------------------


def test_every_task_completes_on_synthetic_data() -> None:
    report = _runner(small_config()).run(write_artifacts=False)
    assert _status(report) == {
        "planned": 0,
        "completed": 10,
        "failed": 0,
        "skipped_budget": 0,
        "skipped_unavailable_dependency": 0,
        "skipped_dependency_failure": 0,
    }
    assert report["results_available"] is True
    assert len(report["rows"]) == 8  # 2 models x 4 strategies
    for row in report["rows"]:
        assert 0.0 <= row["mean_macro_f1"] <= 1.0
        assert row["n_folds"] == 3
        assert row["estimated"]["exact"] is False
        assert row["observed"]["estimator_fits"] == 3


def test_logistic_all_features_matches_the_shared_evaluator() -> None:
    dataset, plan = make_dataset(), make_plan()
    report = _runner(
        tiny_config(), dataset=dataset, plan=plan
    ).run(write_artifacts=False)
    reference = evaluate_feature_subset(dataset, plan, FEATURES)
    task = report["tasks"][0]
    assert task["status"] == STATUS_COMPLETED
    assert task["result"]["mean_macro_f1"] == reference.mean_macro_f1
    assert task["result"]["std_macro_f1"] == reference.std_macro_f1
    assert task["result"]["n_binary_fits"] == reference.n_model_fits
    assert task["result"]["converged"] is reference.converged


def test_every_task_uses_the_same_folds() -> None:
    plan = make_plan()
    report = _runner(small_config(), plan=plan).run(write_artifacts=False)
    expected = [fold.name for fold in plan.folds]
    for task in report["tasks"]:
        if task["kind"] != "evaluation" or task["status"] != STATUS_COMPLETED:
            continue
        assert [fold["fold"] for fold in task["result"]["folds"]] == expected


def test_rare_label_is_excluded_and_reported_by_the_runner() -> None:
    report = _runner(tiny_config()).run(write_artifacts=False)
    result = report["tasks"][0]["result"]
    assert result["excluded_labels"] == ["GO:RARE"]
    assert all("GO:RARE" in fold["excluded_labels"] for fold in result["folds"])


def test_selection_results_are_reused_by_dependent_evaluations() -> None:
    report = _runner(small_config()).run(write_artifacts=False)
    selections = {
        task["strategy"]: task for task in report["tasks"] if task["kind"] == "selection"
    }
    assert selections["forward"]["result"]["selected"]
    for task in report["tasks"]:
        if task["kind"] != "evaluation" or task["strategy"] not in selections:
            continue
        expected = selections[task["strategy"]]["result"]["selected"]
        assert task["result"]["features"] == expected


# ---------------------------------------------------------------------------
# External test block: guarded, not merely corrupted
# ---------------------------------------------------------------------------


def test_runner_hands_a_guarded_dataset_to_the_evaluator(monkeypatch) -> None:
    from protein_database.ml.experiment_guard import TestBlockAccessError

    calls: list[str] = []
    real = runner_module.evaluate_feature_subset

    def spy(dataset, plan, features, **kwargs):
        calls.append(type(dataset.X).__name__)
        with pytest.raises(TestBlockAccessError):
            dataset.X[plan.test]
        with pytest.raises(TestBlockAccessError):
            dataset.y[plan.test]
        return real(dataset, plan, features, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate_feature_subset", spy)
    report = _runner(tiny_config()).run(write_artifacts=False)
    assert calls and set(calls) == {"_GuardedRows"}
    assert _status(report)["completed"] == 1


def test_tampering_with_the_test_block_does_not_change_the_report() -> None:
    plan, dataset = make_plan(), make_dataset()
    clean = _runner(tiny_config(), dataset=dataset, plan=plan).run(write_artifacts=False)

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
    dirty = _runner(tiny_config(), dataset=tampered, plan=plan).run(
        write_artifacts=False
    )
    assert clean["rows"] == dirty["rows"]
    assert clean["tasks"][0]["result"] == dirty["tasks"][0]["result"]
    assert guard_dataset(dataset, plan).X.shape == dataset.X.shape


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_budget_limits_the_number_of_tasks() -> None:
    report = _runner(small_config(), budget=RunBudget(max_tasks=3)).run(
        write_artifacts=False
    )
    assert _status(report)["completed"] == 3
    assert _status(report)["skipped_budget"] == 7
    skipped = [task for task in report["tasks"] if task["status"] == STATUS_SKIPPED_BUDGET]
    assert all("max_tasks" in task["skip_reason"] for task in skipped)
    assert all(task["result"] is None for task in skipped)
    assert report["observed_cost"]["consumed"]["tasks_started"] == 3


def test_budget_limits_fold_evaluations() -> None:
    config = small_config(
        selection=(), strategies=("all_features",), models=small_config().models
    )
    report = _runner(config, budget=RunBudget(max_fold_evaluations=4)).run(
        write_artifacts=False
    )
    assert _status(report)["completed"] == 1
    assert _status(report)["skipped_budget"] == 1
    assert report["observed_cost"]["consumed"]["fold_evaluations"] == 3


def test_selection_budget_skips_dependents_without_inventing_a_subset() -> None:
    report = _runner(
        small_config(),
        budget=RunBudget(max_selection_candidate_evaluations=1),
    ).run(write_artifacts=False)
    assert _status(report)["skipped_budget"] == 2
    assert _status(report)["skipped_dependency_failure"] == 4
    assert _status(report)["completed"] == 4  # all_features + variance_prefilter
    dependents = [
        task
        for task in report["tasks"]
        if task["status"] == STATUS_SKIPPED_DEPENDENCY_FAILURE
    ]
    assert all("did not complete" in task["skip_reason"] for task in dependents)
    assert all(task["result"] is None for task in dependents)
    assert all(task["strategy"] in ("forward", "backward") for task in dependents)


# ---------------------------------------------------------------------------
# Failures, dependencies and convergence
# ---------------------------------------------------------------------------


def test_selection_failure_propagates_to_dependent_tasks(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("forward selection exploded")

    monkeypatch.setattr(runner_module, "run_forward_selection", broken)
    report = _runner(small_config()).run(write_artifacts=False)
    assert _status(report)["failed"] == 1
    assert _status(report)["skipped_dependency_failure"] == 2
    assert _status(report)["completed"] == 7

    failed = [task for task in report["tasks"] if task["status"] == STATUS_FAILED]
    assert failed[0]["error"] == "RuntimeError: forward selection exploded"
    assert failed[0]["result"] is None
    dependents = [
        task
        for task in report["tasks"]
        if task["status"] == STATUS_SKIPPED_DEPENDENCY_FAILURE
    ]
    assert all(task["strategy"] == "forward" for task in dependents)
    assert all(task["result"] is None for task in dependents)


def test_missing_dependency_at_execution_is_recorded(monkeypatch) -> None:
    config = small_config(models=(model_spec("boost", XGBoostConfig()),))
    report = _runner(
        config, dependency_available=lambda module: True
    ).run(write_artifacts=False)
    skipped = [
        task
        for task in report["tasks"]
        if task["status"] == STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY
    ]
    assert len(skipped) == len(config.strategies)
    assert all("xgboost" in task["skip_reason"] for task in skipped)
    assert all(task["result"] is None for task in skipped)
    assert _status(report)["failed"] == 0
    assert _status(report)["completed"] == len(config.selection)


def test_non_convergence_is_reported_and_not_treated_as_an_error() -> None:
    config = tiny_config(models=(model_spec("slow", LogisticConfig(max_iter=1)),))
    report = _runner(config).run(write_artifacts=False)
    assert _status(report)["completed"] == 1
    assert _status(report)["failed"] == 0
    task = report["tasks"][0]
    assert task["result"]["converged"] is False
    assert task["warnings"], "a non-converged fit must carry a warning"
    assert "ConvergenceWarning" in task["warnings"][0]


def test_abort_policy_leaves_remaining_tasks_planned(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner_module, "run_forward_selection", broken)
    config = small_config(failure_policy=FailurePolicy(on_task_error="abort"))
    report = _runner(config).run(write_artifacts=False)
    assert _status(report)["failed"] == 1
    assert _status(report)["planned"] == 9
    assert _status(report)["completed"] == 0
    assert "cancelled" not in report["status_counts"]
    remaining = [task for task in report["tasks"] if task["status"] == STATUS_PLANNED]
    assert all("aborted" in task["note"] for task in remaining)
    assert all(task["result"] is None for task in remaining)
    assert report["results_available"] is False


def test_completed_task_requires_a_result() -> None:
    with pytest.raises(ValueError, match="must carry a result"):
        TaskResult(task_id="t", status=STATUS_COMPLETED)
    with pytest.raises(ValueError, match="must carry an error"):
        TaskResult(task_id="t", status=STATUS_FAILED)
    with pytest.raises(ValueError, match="skip_reason"):
        TaskResult(task_id="t", status=STATUS_SKIPPED_BUDGET)

# ---------------------------------------------------------------------------
# Persistence and resumption
# ---------------------------------------------------------------------------


def _stable(report: dict) -> str:
    """Report text without the metadata that is expected to vary."""
    payload = {key: value for key, value in report.items() if key != "created_at"}
    payload["observed_cost"] = dict(payload["observed_cost"])
    payload["observed_cost"]["consumed"] = {
        key: value
        for key, value in payload["observed_cost"]["consumed"].items()
        if key != "elapsed_seconds"
    }
    return json.dumps(payload, sort_keys=True)


def test_store_and_artifacts_are_written(tmp_path) -> None:
    store = tmp_path / "run.jsonl"
    report = _runner(small_config()).run(store_path=store)

    header, records = TaskStore(store).read()
    assert header is not None
    assert header["record"] == "header"
    assert header["plan_id"] == report["plan_id"]
    assert header["protocol_version"] == report["protocol_version"]
    assert len(records) == 10
    assert {record["status"] for record in records.values()} == {"completed"}

    manifest_file = tmp_path / "run.jsonl.manifest.json"
    report_file = tmp_path / "run.jsonl.report.json"
    manifest = json.loads(manifest_file.read_text())
    written = json.loads(report_file.read_text())
    assert manifest["plan_id"] == report["plan_id"]
    assert written["status_counts"] == report["status_counts"]


def test_resume_reuses_only_completed_tasks(tmp_path, monkeypatch) -> None:
    store = tmp_path / "run.jsonl"
    first = _runner(small_config(), budget=RunBudget(max_tasks=4)).run(store_path=store)
    assert _status(first)["completed"] == 4
    assert _status(first)["skipped_budget"] == 6

    calls = {"evaluation": 0, "selection": 0}
    real_eval = runner_module.evaluate_feature_subset
    real_forward = runner_module.run_forward_selection

    def counting_eval(*args, **kwargs):
        calls["evaluation"] += 1
        return real_eval(*args, **kwargs)

    def counting_forward(*args, **kwargs):
        calls["selection"] += 1
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate_feature_subset", counting_eval)
    monkeypatch.setattr(runner_module, "run_forward_selection", counting_forward)

    second = _runner(small_config()).run(store_path=store, resume=True)
    assert _status(second)["completed"] == 10
    assert calls["selection"] == 0  # both selections were reused
    assert calls["evaluation"] == 6  # 8 evaluation tasks, 2 already stored

    _, records = TaskStore(store).read()
    assert len(records) == 10
    assert {record["status"] for record in records.values()} == {"completed"}


def test_resume_refuses_a_store_from_another_run(tmp_path) -> None:
    store = tmp_path / "run.jsonl"
    _runner(tiny_config()).run(store_path=store)
    with pytest.raises(ResumeMismatchError, match="cannot be reused"):
        _runner(small_config()).run(store_path=store, resume=True)


def test_resume_disabled_starts_a_fresh_store(tmp_path) -> None:
    store = tmp_path / "run.jsonl"
    _runner(tiny_config()).run(store_path=store)
    report = _runner(tiny_config()).run(store_path=store, resume=False)
    assert _status(report)["completed"] == 1
    _, records = TaskStore(store).read()
    assert len(records) == 1


def test_store_rejects_corrupted_content(tmp_path) -> None:
    store = tmp_path / "broken.jsonl"
    store.write_text('{"record": "nonsense"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown record type"):
        TaskStore(store).read()
    store.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        TaskStore(store).read()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_plan_are_equivalent() -> None:
    first = _runner(small_config()).run(write_artifacts=False)
    second = _runner(small_config()).run(write_artifacts=False)
    assert _stable(first) == _stable(second)
    assert first["rows"] == second["rows"]
    assert first["plan_id"] == second["plan_id"]
    assert first["estimated_cost_total"] == second["estimated_cost_total"]


def test_report_cost_keeps_estimates_and_observations_apart() -> None:
    report = _runner(tiny_config()).run(write_artifacts=False)
    task = report["tasks"][0]
    assert task["estimated"]["fold_evaluations"] == 3
    assert task["estimated"]["exact"] is False
    assert task["observed"]["fold_evaluations"] == 3
    assert task["observed"]["tasks"] == 1
    consumed = report["observed_cost"]["consumed"]
    assert consumed["fold_evaluations"] == 3
    assert report["estimated_cost_total"]["tasks"] == 1

