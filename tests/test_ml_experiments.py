"""Behaviour tests for the experimental flow (synthetic data only)."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from protein_database.ml.experiments import (
    ExperimentSettings,
    data_fingerprint,
    development_block,
    evaluate_external,
    run_experiment,
)
from protein_database.ml.feature_selection import (
    eligible_features,
    evaluate_feature_subset,
)
from protein_database.ml.models import LogisticConfig, XGBoostConfig
from protein_database.ml.splits import SplitPlan
from tests.experiment_helpers import make_dataset, make_plan, small_models, small_settings

FAST = dict(strategies=("all_features",), models=(LogisticConfig(max_iter=500),))


# ---------------------------------------------------------------------------
# Development isolation
# ---------------------------------------------------------------------------


def test_development_block_holds_only_development_rows() -> None:
    dataset, plan = make_dataset(), make_plan()

    development, dev_plan = development_block(dataset, plan)

    assert development.n_samples == plan.dev.size
    assert plan.test.size > 0
    test_accessions = {dataset.keys[int(index)].accession for index in plan.test}
    assert test_accessions.isdisjoint(
        {key.accession for key in development.keys}
    )
    assert dev_plan.test.size == 0


def test_development_folds_cover_the_slice_and_stay_disjoint() -> None:
    dataset, plan = make_dataset(), make_plan()

    development, dev_plan = development_block(dataset, plan)

    validated = np.concatenate([fold.validation for fold in dev_plan.folds])
    assert np.array_equal(np.sort(validated), np.arange(development.n_samples))
    for fold in dev_plan.folds:
        assert np.intersect1d(fold.train, fold.validation).size == 0
        assert fold.train.max() < development.n_samples


def test_development_slice_reproduces_the_full_dataset_numbers() -> None:
    """The slice must not change any score: it only removes test rows."""
    dataset, plan = make_dataset(), make_plan()
    development, dev_plan = development_block(dataset, plan)

    full = evaluate_feature_subset(dataset, plan, dataset.feature_names)
    sliced = evaluate_feature_subset(development, dev_plan, development.feature_names)

    assert sliced.mean_macro_f1 == full.mean_macro_f1
    assert sliced.std_macro_f1 == full.std_macro_f1
    assert [fold.macro_f1 for fold in sliced.fold_scores] == [
        fold.macro_f1 for fold in full.fold_scores
    ]


def test_run_experiment_ignores_the_test_values() -> None:
    """Corrupting every test row must not change a single reported number."""
    dataset, plan = make_dataset(), make_plan()
    settings = small_settings()

    clean = run_experiment(dataset, plan, settings)

    poisoned_X = dataset.X.copy()
    poisoned_X[plan.test, :] = 1e6
    poisoned_y = dataset.y.copy()
    poisoned_y[plan.test, :] = True
    poisoned = replace(dataset, X=poisoned_X, y=poisoned_y)

    dirty = run_experiment(poisoned, plan, settings)

    assert json.dumps(clean["rows"], sort_keys=True) == json.dumps(
        dirty["rows"], sort_keys=True
    )
    assert clean["fingerprint"] != dirty["fingerprint"]


# ---------------------------------------------------------------------------
# What the experiment reports
# ---------------------------------------------------------------------------


def test_run_experiment_covers_every_strategy_and_model() -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(dataset, plan, small_settings())

    pairs = {(row["strategy"], row["model_id"]) for row in result["rows"]}
    assert {strategy for strategy, _ in pairs} == {
        "all_features",
        "variance_prefilter",
        "forward",
        "backward",
    }
    assert len(pairs) == 4 * len(small_models())
    for row in result["rows"]:
        assert row["n_folds"] == plan.n_folds
        assert row["error"] is None
        assert 0.0 <= row["mean_macro_f1"] <= 1.0
        assert row["features"]


def test_variance_baseline_is_the_intersection_over_folds() -> None:
    """Baseline 2 keeps what every fold training block keeps; the constant
    column survives only in the all-features baseline."""
    dataset, plan = make_dataset(), make_plan()
    settings = small_settings(
        strategies=("all_features", "variance_prefilter"),
        models=(LogisticConfig(max_iter=500),),
    )

    result = run_experiment(dataset, plan, settings)

    development, dev_plan = development_block(dataset, plan)
    expected = eligible_features(development, dev_plan, config=settings.evaluation)
    rows = {row["strategy"]: row for row in result["rows"]}
    assert tuple(rows["variance_prefilter"]["features"]) == expected
    assert "constant" in rows["all_features"]["features"]
    assert "constant" not in rows["variance_prefilter"]["features"]

def test_every_strategy_uses_the_same_folds() -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(dataset, plan, small_settings())

    fold_names = {
        tuple(fold["fold"] for fold in row["folds"]) for row in result["rows"]
    }
    assert fold_names == {tuple(f"fold_{index}" for index in range(plan.n_folds))}


def test_selection_audit_is_recorded_per_strategy() -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(dataset, plan, small_settings())

    for strategy in ("forward", "backward"):
        audit = result["selections"][strategy]
        assert audit["selected"]
        assert audit["stopped_reason"]
        assert audit["n_evaluations"] > 0
        assert audit["fold_label_coverage"]


def test_table_has_one_line_per_row() -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(dataset, plan, small_settings(**FAST))

    lines = result["table"].splitlines()
    assert len(lines) == len(result["rows"]) + 2
    assert "macro-F1" in lines[0]


# ---------------------------------------------------------------------------
# Optional dependency and settings validation
# ---------------------------------------------------------------------------


def test_missing_xgboost_is_reported_instead_of_substituted() -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(
        dataset,
        plan,
        small_settings(strategies=("all_features",), models=(XGBoostConfig(),)),
    )

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["model_id"] == "xgboost_ovr"
    assert row["mean_macro_f1"] is None
    assert "xgboost" in row["error"]


def test_settings_reject_two_configurations_of_one_family() -> None:
    with pytest.raises(ValueError, match="same family"):
        ExperimentSettings(models=(LogisticConfig(), LogisticConfig(C=0.5)))


def test_settings_reject_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        ExperimentSettings(strategies=("all_features", "magic"))


def test_settings_reject_an_empty_comparison() -> None:
    with pytest.raises(ValueError, match="strategies is empty"):
        ExperimentSettings(strategies=())
    with pytest.raises(ValueError, match="models is empty"):
        ExperimentSettings(models=())


# ---------------------------------------------------------------------------
# Stored results and resume
# ---------------------------------------------------------------------------


def test_results_are_written_as_json_and_csv(tmp_path: Path) -> None:
    dataset, plan = make_dataset(), make_plan()

    result = run_experiment(
        dataset, plan, small_settings(**FAST), out_dir=tmp_path, sources={"db": "x.duckdb"}
    )

    stored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert stored["fingerprint"] == result["fingerprint"]
    assert stored["sources"] == {"db": "x.duckdb"}
    assert stored["data"]["n_test"] == int(plan.test.size)

    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result["rows"])
    assert rows[0]["strategy"] == "all_features"


def test_resume_reuses_stored_rows_when_the_data_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    dataset, plan = make_dataset(), make_plan()
    settings = small_settings(**FAST)
    run_experiment(dataset, plan, settings, out_dir=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("a resumed run must not score anything again")

    monkeypatch.setattr("protein_database.ml.experiments.compare_models", forbidden)
    again = run_experiment(dataset, plan, settings, out_dir=tmp_path, resume=True)

    assert len(again["rows"]) == 1


def test_changing_the_labels_invalidates_stored_rows(tmp_path: Path) -> None:
    dataset, plan = make_dataset(), make_plan()
    settings = small_settings(**FAST)
    first = run_experiment(dataset, plan, settings, out_dir=tmp_path)

    y = dataset.y.copy()
    y[0, 0] = not y[0, 0]
    changed = replace(dataset, y=y)
    second = run_experiment(changed, plan, settings, out_dir=tmp_path, resume=True)

    assert first["fingerprint"] != second["fingerprint"]
    stored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert stored["fingerprint"] == second["fingerprint"]


def test_changing_the_settings_invalidates_the_fingerprint() -> None:
    dataset, plan = make_dataset(), make_plan()

    first = data_fingerprint(dataset, small_settings())
    second = data_fingerprint(dataset, small_settings(seed=7))

    assert first != second


# ---------------------------------------------------------------------------
# External evaluation
# ---------------------------------------------------------------------------


def test_evaluate_external_scores_the_test_block_once() -> None:
    dataset, plan = make_dataset(), make_plan()

    report = evaluate_external(
        dataset, plan, dataset.feature_names, LogisticConfig(max_iter=500)
    )

    assert report["n_test_samples"] == int(plan.test.size)
    assert report["n_fit_samples"] == int(plan.dev.size)
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["n_labels_evaluated"] >= 1
    assert report["model"]["model_id"] == "logistic_ovr"


def test_evaluate_external_requires_a_test_block() -> None:
    dataset, plan = make_dataset(), make_plan()
    _development, dev_plan = development_block(dataset, plan)

    assert isinstance(dev_plan, SplitPlan)
    with pytest.raises(ValueError, match="no external test row"):
        evaluate_external(_development, dev_plan, _development.feature_names)


def test_evaluate_external_rejects_an_unknown_feature() -> None:
    dataset, plan = make_dataset(), make_plan()

    with pytest.raises(ValueError):
        evaluate_external(dataset, plan, ["does_not_exist"])


def test_external_evaluation_is_not_part_of_run_experiment(
    monkeypatch, tmp_path: Path
) -> None:
    dataset, plan = make_dataset(), make_plan()

    def forbidden(*args, **kwargs):
        raise AssertionError("run_experiment must not evaluate the external block")

    monkeypatch.setattr("protein_database.ml.experiments.evaluate_external", forbidden)
    result = run_experiment(dataset, plan, small_settings(**FAST), out_dir=tmp_path)

    assert result["rows"]
