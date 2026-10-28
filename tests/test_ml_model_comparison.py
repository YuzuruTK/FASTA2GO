"""Tests for the model comparison layer (synthetic data only, no real base)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml import feature_selection as selection_module
from protein_database.ml import models as models_module
from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.feature_selection import (
    ForwardSelectionConfig,
    eligible_features,
    evaluate_feature_subset,
    run_backward_elimination,
    run_forward_selection,
)
from protein_database.ml.model_comparison import (
    INHERENT_LIMITS_NOTE,
    NO_WINNER_NOTE,
    ModelComparison,
    ModelEvaluation,
    compare_models,
)
from protein_database.ml.models import (
    LogisticConfig,
    MissingDependencyError,
    RandomForestConfig,
    XGBoostConfig,
)
from protein_database.ml.splits import make_group_split_plan
from protein_database.ml.strategies import (
    STRATEGY_ALL,
    STRATEGY_BACKWARD,
    STRATEGY_FORWARD,
    STRATEGY_VARIANCE,
)

GROUPS = np.repeat(np.arange(12), 6)
LABELS = ("GO:0003824", "GO:0005488", "GO:RARE")
FEATURES = ("f_a", "n1", "n2", "constant")
LOGISTIC = LogisticConfig(max_iter=500)
FOREST = RandomForestConfig(n_estimators=20)


def _dataset(seed: int = 3) -> ProteinDataset:
    """Two learnable labels, one label with a single positive, one constant column."""
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 2))
    rare = np.zeros(n, dtype=bool)
    rare[0] = True
    return ProteinDataset(
        keys=tuple(
            ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
            for index in range(n)
        ),
        X=np.column_stack([signal, noise[:, 0], noise[:, 1], np.ones(n)]),
        y=np.column_stack([signal > 0, signal > 0.5, rare]),
        feature_names=FEATURES,
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )


def _plan(seed: int = 3):
    return make_group_split_plan(GROUPS, seed=seed)


def _subsets(dataset: ProteinDataset, plan) -> dict[str, tuple[str, ...]]:
    """The four strategies, built once and then held fixed per model."""
    config = ForwardSelectionConfig(max_features=2)
    forward = run_forward_selection(dataset, plan, config=config)
    backward = run_backward_elimination(dataset, plan, config=config)
    return {
        STRATEGY_ALL: dataset.feature_names,
        STRATEGY_VARIANCE: eligible_features(dataset, plan, config=config),
        STRATEGY_FORWARD: forward.selection.selected,
        STRATEGY_BACKWARD: backward.remaining,
    }


def _comparison(dataset=None, plan=None, **kwargs) -> ModelComparison:
    dataset = dataset if dataset is not None else _dataset()
    plan = plan if plan is not None else _plan()
    return compare_models(
        dataset,
        plan,
        subsets=kwargs.pop("subsets", None) or _subsets(dataset, plan),
        models=kwargs.pop("models", (LOGISTIC, FOREST)),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Subset plumbing
# ---------------------------------------------------------------------------


def test_the_four_strategies_give_four_distinct_subsets() -> None:
    dataset, plan = _dataset(), _plan()
    subsets = _subsets(dataset, plan)
    assert list(subsets) == [
        STRATEGY_ALL,
        STRATEGY_VARIANCE,
        STRATEGY_FORWARD,
        STRATEGY_BACKWARD,
    ]
    assert subsets[STRATEGY_ALL] == dataset.feature_names
    # the constant column is removed by the variance filter, never by the
    # all-features baseline
    assert "constant" not in subsets[STRATEGY_VARIANCE]
    assert "constant" in subsets[STRATEGY_ALL]


# ---------------------------------------------------------------------------
# Comparison matrix
# ---------------------------------------------------------------------------


def test_every_model_is_scored_on_every_subset() -> None:
    comparison = _comparison()
    assert comparison.model_ids == ("logistic_ovr", "random_forest_ovr")
    assert len(comparison.subset_labels) == 4
    assert len(comparison.evaluations) == 8
    assert comparison.failures() == ()
    for evaluation in comparison.evaluations:
        assert 0.0 <= evaluation.mean_macro_f1 <= 1.0
        assert evaluation.n_features == len(evaluation.features)
        assert evaluation.std_macro_f1 >= 0.0


def test_all_models_use_exactly_the_same_folds() -> None:
    dataset, plan = _dataset(), _plan()
    comparison = _comparison(dataset, plan)
    expected = tuple(fold.name for fold in plan.folds)
    assert comparison.folds == expected
    for evaluation in comparison.evaluations:
        assert tuple(fold.fold for fold in evaluation.fold_scores) == expected
        assert evaluation.n_estimator_fits == len(expected)


def test_subset_features_are_recorded_verbatim() -> None:
    comparison = _comparison()
    for label, features in comparison.subsets.items():
        for model_id in comparison.model_ids:
            assert comparison.get(model_id, label).features == features


def test_logistic_scores_match_the_phase_seven_evaluator_exactly() -> None:
    """The shared evaluator must not change a single score for the reference model."""
    dataset, plan = _dataset(), _plan()
    subsets = _subsets(dataset, plan)
    comparison = _comparison(dataset, plan, subsets=subsets)
    for label, features in subsets.items():
        reference = evaluate_feature_subset(dataset, plan, features)
        row = comparison.get("logistic_ovr", label)
        assert row.mean_macro_f1 == reference.mean_macro_f1
        assert row.std_macro_f1 == reference.std_macro_f1
        assert row.n_binary_fits == reference.n_model_fits


def test_fit_counters_separate_pipelines_from_binary_models() -> None:
    comparison = _comparison()
    for evaluation in comparison.evaluations:
        assert evaluation.n_estimator_fits == len(evaluation.fold_scores)
        # OvR trains one binary model per evaluable label, so this can be larger
        assert evaluation.n_binary_fits >= evaluation.n_estimator_fits


def test_random_forest_is_deterministic_across_runs() -> None:
    dataset, plan = _dataset(), _plan()
    subsets = {STRATEGY_ALL: dataset.feature_names}
    first = _comparison(dataset, plan, subsets=subsets, models=(FOREST,))
    second = _comparison(dataset, plan, subsets=subsets, models=(FOREST,))
    assert first.rows() == second.rows()


def test_rare_label_is_excluded_and_reported() -> None:
    comparison = _comparison()
    evaluation = comparison.get("logistic_ovr", STRATEGY_ALL)
    assert evaluation.excluded_labels == ("GO:RARE",)
    for fold in evaluation.fold_scores:
        assert "GO:RARE" in fold.excluded_labels
        assert fold.n_labels_evaluated < fold.n_labels_total
    assert "GO:RARE" in evaluation.as_dict()["folds"][0]["excluded_labels"]


# ---------------------------------------------------------------------------
# Leakage: the external test block must never be touched
# ---------------------------------------------------------------------------


def test_external_test_tampering_does_not_change_the_comparison() -> None:
    plan = _plan()
    dataset = _dataset()
    clean = _comparison(dataset, plan)

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
    dirty = _comparison(tampered, plan)

    assert clean.rows() == dirty.rows()
    for before, after in zip(clean.evaluations, dirty.evaluations):
        assert before.excluded_labels == after.excluded_labels
        assert [fold.macro_f1 for fold in before.fold_scores] == [
            fold.macro_f1 for fold in after.fold_scores
        ]


def test_test_indices_are_never_read_by_the_comparison(monkeypatch) -> None:
    """Fail loudly if anything slices the dataset with the test indices."""
    plan = _plan()
    dataset = _dataset()
    real_build = selection_module._build_contexts
    seen: list[tuple[int, ...]] = []

    def spy(dataset_arg, plan_arg, config, **kwargs):
        frames = real_build(dataset_arg, plan_arg, config, **kwargs)
        for _, fold_plan in enumerate(plan_arg.folds):
            seen.append(tuple(sorted(fold_plan.validation)))
        return frames

    monkeypatch.setattr(selection_module, "_build_contexts", spy)
    _comparison(dataset, plan, subsets={STRATEGY_ALL: dataset.feature_names})
    test_set = set(plan.test.tolist())
    for fold in plan.folds:
        assert not test_set.intersection(fold.train.tolist())
        assert not test_set.intersection(fold.validation.tolist())
    assert seen  # the spy really ran


# ---------------------------------------------------------------------------
# Failures and non-convergence
# ---------------------------------------------------------------------------


def test_missing_optional_dependency_is_recorded_not_raised(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "xgboost_available", lambda: False)
    dataset, plan = _dataset(), _plan()
    comparison = compare_models(
        dataset,
        plan,
        subsets={STRATEGY_ALL: dataset.feature_names},
        models=(LOGISTIC, XGBoostConfig(n_estimators=5)),
    )
    failed = comparison.failures()
    assert len(failed) == 1
    assert failed[0].model_id == "xgboost_ovr"
    assert "xgboost" in failed[0].error
    assert failed[0].mean_macro_f1 is None
    assert comparison.get("logistic_ovr", STRATEGY_ALL).failed is False


def test_on_error_raise_propagates_the_dependency_error(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "xgboost_available", lambda: False)
    dataset, plan = _dataset(), _plan()
    with pytest.raises(MissingDependencyError):
        compare_models(
            dataset,
            plan,
            subsets={STRATEGY_ALL: dataset.feature_names},
            models=(XGBoostConfig(n_estimators=5),),
            on_error="raise",
        )


def test_estimator_crash_is_recorded_with_its_error(monkeypatch) -> None:
    def explode(config=None):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(selection_module, "build_estimator", explode)
    dataset, plan = _dataset(), _plan()
    comparison = compare_models(
        dataset,
        plan,
        subsets={STRATEGY_ALL: dataset.feature_names},
        models=(FOREST,),
    )
    (failed,) = comparison.failures()
    assert failed.error == "RuntimeError: estimator exploded"
    assert failed.n_estimator_fits == 0
    assert failed.n_binary_fits == 0


def test_non_convergent_estimator_is_flagged_not_swallowed() -> None:
    dataset, plan = _dataset(), _plan()
    comparison = compare_models(
        dataset,
        plan,
        subsets={STRATEGY_ALL: dataset.feature_names},
        models=(LogisticConfig(max_iter=1),),
    )
    evaluation = comparison.get("logistic_ovr", STRATEGY_ALL)
    assert evaluation.converged is False
    assert evaluation.mean_macro_f1 is not None
    assert comparison.non_converged() == (evaluation,)
    assert evaluation.as_dict()["converged"] is False


# ---------------------------------------------------------------------------
# Validation of inputs
# ---------------------------------------------------------------------------


def test_unknown_feature_is_rejected() -> None:
    dataset, plan = _dataset(), _plan()
    with pytest.raises(ValueError, match="unknown feature"):
        compare_models(
            dataset,
            plan,
            subsets={STRATEGY_ALL: ("f_a", "does_not_exist")},
            models=(LOGISTIC,),
        )


def test_empty_and_repeated_subsets_are_rejected() -> None:
    dataset, plan = _dataset(), _plan()
    with pytest.raises(ValueError, match="empty"):
        compare_models(dataset, plan, subsets={}, models=(LOGISTIC,))
    with pytest.raises(ValueError, match="empty"):
        compare_models(dataset, plan, subsets={STRATEGY_ALL: ()}, models=(LOGISTIC,))
    with pytest.raises(ValueError, match="repeats"):
        compare_models(
            dataset, plan, subsets={STRATEGY_ALL: ("f_a", "f_a")}, models=(LOGISTIC,)
        )


def test_models_must_be_supported_and_unique() -> None:
    dataset, plan = _dataset(), _plan()
    subsets = {STRATEGY_ALL: dataset.feature_names}
    with pytest.raises(ValueError, match="models is empty"):
        compare_models(dataset, plan, subsets=subsets, models=())
    with pytest.raises(TypeError, match="Unsupported model configuration"):
        compare_models(dataset, plan, subsets=subsets, models=("logistic",))
    with pytest.raises(ValueError, match="Duplicate model_id"):
        compare_models(
            dataset, plan, subsets=subsets, models=(LOGISTIC, LogisticConfig(C=0.5))
        )
    with pytest.raises(ValueError, match="on_error"):
        compare_models(dataset, plan, subsets=subsets, models=(LOGISTIC,), on_error="x")


def test_failed_evaluation_requires_an_error_message() -> None:
    with pytest.raises(ValueError, match="error message"):
        ModelEvaluation(
            model_id="logistic_ovr",
            model={},
            subset="all_features",
            features=("f_a",),
            evaluation=None,
        )


def test_get_requires_an_existing_unique_pair() -> None:
    comparison = _comparison()
    with pytest.raises(KeyError):
        comparison.get("xgboost_ovr", STRATEGY_ALL)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_is_json_serialisable_and_labelled_as_synthetic() -> None:
    comparison = _comparison()
    payload = comparison.as_dict()
    assert json.loads(comparison.to_json()) == payload
    assert INHERENT_LIMITS_NOTE in payload["notes"]
    assert NO_WINNER_NOTE in payload["notes"]
    assert payload["policy"]["external_test"] == "never used"
    assert payload["policy"]["threshold"] == 0.5
    row = payload["rows"][0]
    assert {"model_id", "subset", "n_features"} <= set(row)
    assert {"mean_macro_f1", "std_macro_f1"} <= set(row)
    assert {"n_estimator_fits", "n_binary_fits"} <= set(row)
    model = payload["models"][0]
    assert model["model_id"] == "logistic_ovr"
    assert model["configuration"]["type"] == "OneVsRestClassifier(LogisticRegression)"


def test_report_does_not_rank_models() -> None:
    comparison = _comparison()
    public = [name for name in dir(comparison) if not name.startswith("_")]
    assert not [name for name in public if "best" in name or "winner" in name]
    assert set(comparison.as_dict()) == {
        "policy",
        "notes",
        "subsets",
        "folds",
        "models",
        "evaluations",
        "rows",
        "failures",
    }


def test_notes_can_be_replaced_by_the_caller() -> None:
    comparison = _comparison(notes=["dados sintéticos apenas"])
    assert comparison.notes == ("dados sintéticos apenas",)
    assert comparison.as_dict()["notes"] == ["dados sintéticos apenas"]
