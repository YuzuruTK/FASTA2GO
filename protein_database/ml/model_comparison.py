"""Compare model families on fixed feature subsets under one evaluation policy.

The design goal is isolation: to compare models, the feature subset must stay
fixed, and to compare subsets, the model must stay fixed. Nothing here searches
features per model - the subsets come from the Phase 7 strategies
(:func:`subsets_from_outcomes`) and each (model, subset) pair is scored with
:func:`protein_database.ml.feature_selection.evaluate_feature_subset`, which is
the same code path used by Forward Selection and Backward Elimination:

* same ``SplitPlan`` (grouped validation folds);
* same label masks (``fit``/``eval`` intersection per fold);
* same ``multilabel_metrics`` macro-F1 and the same 0.5 threshold;
* any transformation is fitted inside the fold's training block;
* ``plan.test`` is never indexed;
* parameters that are provisional are marked as such in the report notes.

Cost accounting keeps two separate counters, because they measure different
things: ``n_estimator_fits`` counts pipelines fitted (one per fold) while
``n_binary_fits`` counts the binary classifiers those pipelines trained
internally (one per evaluable label per fold under OvR).

The report records numbers; it does not rank models. A difference in mean
macro-F1 on synthetic data says nothing about scientific quality, and no
significance test is implemented here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from protein_database.ml.dataset import ProteinDataset, validate_dataset
from protein_database.ml.feature_selection import (
    FoldScore,
    ForwardSelectionConfig,
    SubsetEvaluation,
    evaluate_feature_subset,
)
from protein_database.ml.models import SUPPORTED_MODEL_CONFIGS, describe_estimator
from protein_database.ml.splits import SplitPlan
from protein_database.ml.strategies import StrategyOutcome

INHERENT_LIMITS_NOTE = (
    "Synthetic runs validate the infrastructure only: they are not evidence "
    "that any model family predicts GO terms better on the real base."
)
NO_WINNER_NOTE = (
    "No winner is declared: differences are not statistically tested and no "
    "threshold for a meaningful gain is defined yet."
)
FIXED_SUBSET_NOTE = (
    "Feature subsets are fixed across models (no per-model feature search); "
    "variance filtering, when present, came from the Phase 7 strategy."
)
DEFAULT_NOTES = (FIXED_SUBSET_NOTE, INHERENT_LIMITS_NOTE, NO_WINNER_NOTE)

ON_ERROR_RECORD = "record"
ON_ERROR_RAISE = "raise"
ON_ERROR_MODES = (ON_ERROR_RECORD, ON_ERROR_RAISE)


def subsets_from_outcomes(
    outcomes: Sequence[StrategyOutcome],
) -> dict[str, tuple[str, ...]]:
    """Turn Phase 7 strategy outcomes into ``{subset_label: features}``.

    The strategies already guarantee that each final subset was scored through
    :func:`evaluate_feature_subset`, so reusing their ``selected`` tuples keeps
    the feature sets identical while the model varies.
    """
    subsets: dict[str, tuple[str, ...]] = {}
    for outcome in outcomes:
        if outcome.strategy in subsets:
            raise ValueError(f"Duplicate strategy {outcome.strategy!r} in outcomes.")
        if not outcome.selected:
            raise ValueError(f"Strategy {outcome.strategy!r} has an empty subset.")
        subsets[outcome.strategy] = tuple(outcome.selected)
    if not subsets:
        raise ValueError("outcomes is empty: nothing to compare.")
    return subsets


def _validate_subsets(
    subsets: Mapping[str, Sequence[str]], dataset: ProteinDataset
) -> dict[str, tuple[str, ...]]:
    if not subsets:
        raise ValueError("subsets is empty: nothing to compare.")
    known = set(dataset.feature_names)
    validated: dict[str, tuple[str, ...]] = {}
    for label, features in subsets.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"Subset label must be a non-empty string; got {label!r}.")
        names = tuple(features)
        if not names:
            raise ValueError(f"Subset {label!r} is empty.")
        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"Subset {label!r} repeats feature(s): {repeated[:5]}.")
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                f"Subset {label!r} has unknown feature(s): {unknown[:5]}."
            )
        validated[label] = names
    return validated


def _fold_rows(fold_scores: Sequence[FoldScore]) -> list[dict]:
    return [
        {
            "fold": score.fold,
            "macro_f1": score.macro_f1,
            "n_labels_evaluated": score.n_labels_evaluated,
            "n_labels_total": score.n_labels_total,
            "n_fit_samples": score.n_fit_samples,
            "n_validation_samples": score.n_validation_samples,
            "excluded_labels": list(score.excluded_labels),
            "converged": score.converged,
        }
        for score in fold_scores
    ]


@dataclass(frozen=True)
class ModelEvaluation:
    """One (model, feature subset) pair scored on the shared folds.

    ``evaluation`` is ``None`` only when the pair could not be scored at all, in
    which case ``error`` explains why (missing optional dependency, estimator
    failure). Non-convergence is *not* an error: it is reported through
    ``converged`` and the offending folds, exactly as the selection code does.
    """

    model_id: str
    model: Mapping[str, object]
    subset: str
    features: tuple[str, ...]
    evaluation: SubsetEvaluation | None
    n_estimator_fits: int = 0
    n_binary_fits: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.evaluation is None and not self.error:
            raise ValueError("A failed evaluation must record an error message.")
        if self.evaluation is not None and self.error:
            raise ValueError("A scored evaluation cannot carry an error message.")

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def mean_macro_f1(self) -> float | None:
        return None if self.evaluation is None else self.evaluation.mean_macro_f1

    @property
    def std_macro_f1(self) -> float | None:
        return None if self.evaluation is None else self.evaluation.std_macro_f1

    @property
    def converged(self) -> bool | None:
        return None if self.evaluation is None else self.evaluation.converged

    @property
    def failed(self) -> bool:
        return self.evaluation is None

    @property
    def excluded_labels(self) -> tuple[str, ...]:
        if self.evaluation is None:
            return ()
        excluded = {
            label
            for fold in self.evaluation.fold_scores
            for label in fold.excluded_labels
        }
        return tuple(sorted(excluded))

    @property
    def fold_scores(self) -> tuple[FoldScore, ...]:
        return () if self.evaluation is None else self.evaluation.fold_scores

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model": dict(self.model),
            "subset": self.subset,
            "n_features": self.n_features,
            "features": list(self.features),
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "n_folds": len(self.fold_scores),
            "folds": _fold_rows(self.fold_scores),
            "excluded_labels": list(self.excluded_labels),
            "converged": self.converged,
            "n_estimator_fits": self.n_estimator_fits,
            "n_binary_fits": self.n_binary_fits,
            "error": self.error,
        }


def _evaluation_policy() -> dict:
    """How every number in the report was produced (no silent conventions)."""
    return {
        "metric": "macro_f1",
        "threshold": 0.5,
        "aggregation": "mean_over_folds",
        "dispersion": "population_std_over_folds",
        "folds": "grouped_validation_folds_of_the_split_plan",
        "label_masks": "fit and eval intersection, identical to the selection code",
        "variance_filter": "not applied at evaluation time",
        "external_test": "never used",
    }


@dataclass(frozen=True)
class ModelComparison:
    """Every (model, subset) pair scored under one policy.

    The object is a report, not a ranking: :meth:`rows` is ordered exactly as the
    subsets and models were given, and no method declares a winner.
    """

    evaluations: tuple[ModelEvaluation, ...]
    subsets: Mapping[str, tuple[str, ...]]
    notes: tuple[str, ...] = DEFAULT_NOTES
    policy: Mapping[str, object] = field(default_factory=_evaluation_policy)

    @property
    def model_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for evaluation in self.evaluations:
            if evaluation.model_id not in seen:
                seen.append(evaluation.model_id)
        return tuple(seen)

    @property
    def subset_labels(self) -> tuple[str, ...]:
        return tuple(self.subsets)

    @property
    def folds(self) -> tuple[str, ...]:
        for evaluation in self.evaluations:
            if evaluation.fold_scores:
                return tuple(score.fold for score in evaluation.fold_scores)
        return ()

    def failures(self) -> tuple[ModelEvaluation, ...]:
        return tuple(evaluation for evaluation in self.evaluations if evaluation.failed)

    def non_converged(self) -> tuple[ModelEvaluation, ...]:
        return tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.converged is False
        )

    def get(self, model_id: str, subset: str) -> ModelEvaluation:
        """The pair ``(model_id, subset)``; keys must exist exactly once."""
        matches = [
            evaluation
            for evaluation in self.evaluations
            if evaluation.model_id == model_id and evaluation.subset == subset
        ]
        if len(matches) != 1:
            raise KeyError(f"No unique evaluation for {model_id!r} / {subset!r}.")
        return matches[0]

    def rows(self) -> list[dict]:
        """Compact table: one row per (subset, model)."""
        return [
            {
                "model_id": evaluation.model_id,
                "subset": evaluation.subset,
                "n_features": evaluation.n_features,
                "mean_macro_f1": evaluation.mean_macro_f1,
                "std_macro_f1": evaluation.std_macro_f1,
                "n_folds": len(evaluation.fold_scores),
                "converged": evaluation.converged,
                "n_estimator_fits": evaluation.n_estimator_fits,
                "n_binary_fits": evaluation.n_binary_fits,
                "error": evaluation.error,
            }
            for evaluation in self.evaluations
        ]

    def as_dict(self) -> dict:
        return {
            "policy": dict(self.policy),
            "notes": list(self.notes),
            "subsets": {
                label: {"n_features": len(features), "features": list(features)}
                for label, features in self.subsets.items()
            },
            "folds": list(self.folds),
            "models": [
                {
                    "model_id": model_id,
                    "configuration": dict(
                        next(
                            item.model
                            for item in self.evaluations
                            if item.model_id == model_id
                        )
                    ),
                }
                for model_id in self.model_ids
            ],
            "evaluations": [evaluation.as_dict() for evaluation in self.evaluations],
            "rows": self.rows(),
            "failures": [
                {"model_id": item.model_id, "subset": item.subset, "error": item.error}
                for item in self.failures()
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON-ready serialisation of the whole report."""
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)


def _check_same_folds(evaluations: Sequence[ModelEvaluation]) -> tuple[str, ...]:
    """Every scored pair must have been measured on the same folds, in order."""
    reference: tuple[str, ...] | None = None
    for evaluation in evaluations:
        if not evaluation.fold_scores:
            continue
        folds = tuple(score.fold for score in evaluation.fold_scores)
        if reference is None:
            reference = folds
        elif folds != reference:
            raise ValueError(
                "Evaluations disagree on the folds: "
                f"{reference} vs {folds} for {evaluation.model_id!r} / "
                f"{evaluation.subset!r}."
            )
    return reference or ()


def compare_models(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    subsets: Mapping[str, Sequence[str]],
    models: Sequence[object],
    config: ForwardSelectionConfig | None = None,
    on_error: str = ON_ERROR_RECORD,
    notes: Sequence[str] | None = None,
) -> ModelComparison:
    """Score every model on every fixed subset with one evaluation policy.

    Args:
        dataset: Aligned features and labels; validated before anything runs.
        plan: The shared ``SplitPlan``. Only its validation folds are used;
            ``plan.test`` is never indexed.
        subsets: ``{label: feature names}``, typically from
            :func:`subsets_from_outcomes`. Kept fixed across models.
        models: Model configurations (:class:`LogisticConfig`,
            :class:`RandomForestConfig`, :class:`XGBoostConfig`). Two
            configurations of the same family are rejected because they would
            share a ``model_id`` in the report.
        config: Evaluation settings. Its ``estimator`` is overridden by each
            model; masks, folds and aggregation stay identical.
        on_error: ``"record"`` (default) keeps a failed pair in the report with
            its error message; ``"raise"`` propagates instead.
        notes: Free-text caveats stored in the report. Defaults to the
            synthetic-data / no-winner notes.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: If subsets, models or ``on_error`` are invalid, or if the
            evaluations disagree on folds.
        TypeError: If a model object is not a supported configuration.
    """
    settings = config or ForwardSelectionConfig()
    if on_error not in ON_ERROR_MODES:
        raise ValueError(f"on_error must be one of {ON_ERROR_MODES}; got {on_error!r}.")
    validate_dataset(dataset)

    validated = _validate_subsets(subsets, dataset)
    model_list = tuple(models)
    if not model_list:
        raise ValueError("models is empty: nothing to compare.")
    for model in model_list:
        if not isinstance(model, SUPPORTED_MODEL_CONFIGS):
            raise TypeError(
                "Unsupported model configuration: "
                f"{type(model).__name__}. Expected one of "
                f"{[cls.__name__ for cls in SUPPORTED_MODEL_CONFIGS]}."
            )
    model_ids = [model.model_id for model in model_list]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError(
            f"Duplicate model_id in models: {model_ids}. Two configurations of "
            "the same family would be indistinguishable in the report."
        )

    evaluations: list[ModelEvaluation] = []
    for label, features in validated.items():
        for model in model_list:
            descriptor = describe_estimator(model)
            try:
                evaluation = evaluate_feature_subset(
                    dataset, plan, features, config=settings, model=model
                )
            except Exception as error:  # recorded by design when on_error="record"
                if on_error == ON_ERROR_RAISE:
                    raise
                evaluations.append(
                    ModelEvaluation(
                        model_id=model.model_id,
                        model=descriptor,
                        subset=label,
                        features=features,
                        evaluation=None,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            evaluations.append(
                ModelEvaluation(
                    model_id=model.model_id,
                    model=descriptor,
                    subset=label,
                    features=features,
                    evaluation=evaluation,
                    n_estimator_fits=len(evaluation.fold_scores),
                    n_binary_fits=evaluation.n_model_fits,
                )
            )

    comparison = ModelComparison(
        evaluations=tuple(evaluations),
        subsets=validated,
        notes=tuple(DEFAULT_NOTES if notes is None else notes),
        policy=_evaluation_policy(),
    )
    _check_same_folds(comparison.evaluations)
    return comparison
