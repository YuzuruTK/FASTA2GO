"""One comparable representation for every feature-selection strategy.

The four strategies are:

1. ``all_features`` - every column, no selection and no variance filtering;
2. ``variance_prefilter`` - the columns kept by the variance filter of every
   inner fold, without any supervised search;
3. ``forward`` - :func:`protein_database.ml.feature_selection.run_forward_selection`;
4. ``backward`` - :func:`protein_database.ml.feature_selection.run_backward_elimination`.

All four are measured with the **same** ``SplitPlan`` (same grouped folds), the
same label masks, the same estimator and the same mean-aggregation, and every
final subset is re-scored through :func:`evaluate_feature_subset` so the numbers
are apples-to-apples. The external test block is never read.

None of this trains a model on real data: the caller supplies a
``ProteinDataset`` and a ``SplitPlan``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from protein_database.ml.dataset import ProteinDataset
from protein_database.ml.feature_selection import (
    BackwardEliminationResult,
    ForwardSelectionConfig,
    ForwardSelectionResult,
    SubsetEvaluation,
    eligible_features,
    evaluate_feature_subset,
)
from protein_database.ml.splits import SplitPlan

STRATEGY_ALL = "all_features"
STRATEGY_VARIANCE = "variance_prefilter"
STRATEGY_FORWARD = "forward"
STRATEGY_BACKWARD = "backward"

STRATEGIES = (STRATEGY_ALL, STRATEGY_VARIANCE, STRATEGY_FORWARD, STRATEGY_BACKWARD)


@dataclass(frozen=True)
class StrategyOutcome:
    """Final subset of one strategy plus its common evaluation and its cost."""

    strategy: str
    selected: tuple[str, ...]
    evaluation: SubsetEvaluation
    n_evaluations: int
    n_model_fits: int
    stopped_reason: str
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.selected)

    @property
    def mean_macro_f1(self) -> float:
        return self.evaluation.mean_macro_f1

    @property
    def std_macro_f1(self) -> float:
        return self.evaluation.std_macro_f1

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_features": self.n_features,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "n_evaluations": self.n_evaluations,
            "n_model_fits": self.n_model_fits,
            "n_model_fits_evaluation": self.evaluation.n_model_fits,
            "stopped_reason": self.stopped_reason,
            "selected": list(self.selected),
            "excluded_folds": list(self.evaluation.excluded_folds),
            "detail": dict(self.detail),
        }


def all_features_outcome(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> StrategyOutcome:
    """Baseline 1: every feature, no selection, no variance filtering."""
    settings = config or ForwardSelectionConfig()
    evaluation = evaluate_feature_subset(
        dataset, plan, dataset.feature_names, config=settings
    )
    return StrategyOutcome(
        strategy=STRATEGY_ALL,
        selected=dataset.feature_names,
        evaluation=evaluation,
        n_evaluations=0,
        n_model_fits=0,
        stopped_reason="not_applicable",
        detail={"note": "no search and no variance filter: every column is kept"},
    )


def variance_prefilter_outcome(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> StrategyOutcome:
    """Baseline 2: the variance filter of every fold, no supervised search."""
    settings = config or ForwardSelectionConfig()
    kept = eligible_features(dataset, plan, config=settings)
    dropped = tuple(name for name in dataset.feature_names if name not in set(kept))
    if not kept:
        raise ValueError(
            "The variance filter removed every feature; there is nothing to "
            "evaluate for this baseline."
        )
    evaluation = evaluate_feature_subset(dataset, plan, kept, config=settings)
    return StrategyOutcome(
        strategy=STRATEGY_VARIANCE,
        selected=kept,
        evaluation=evaluation,
        n_evaluations=0,
        n_model_fits=0,
        stopped_reason="not_applicable",
        detail={
            "dropped_by_variance_filter": list(dropped),
            "variance_threshold": settings.variance_threshold,
        },
    )


def forward_outcome(
    result: ForwardSelectionResult,
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> StrategyOutcome:
    """Wrap a forward run, re-scoring its final subset on the common basis."""
    settings = config or result.config
    selected = result.selection.selected
    if not selected:
        raise ValueError("The forward run selected no feature; nothing to compare.")
    return StrategyOutcome(
        strategy=STRATEGY_FORWARD,
        selected=selected,
        evaluation=evaluate_feature_subset(dataset, plan, selected, config=settings),
        n_evaluations=result.selection.n_evaluations,
        n_model_fits=result.selection.n_model_fits,
        stopped_reason=result.selection.stopped_reason,
        detail={
            "search_best_score": result.selection.best_score,
            "prefilter_dropped": list(result.prefilter_dropped),
            "n_rounds": len(result.selection.history),
        },
    )


def backward_outcome(
    result: BackwardEliminationResult,
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> StrategyOutcome:
    """Wrap a backward run, re-scoring its final subset on the common basis."""
    settings = config or result.config
    if not result.remaining:
        raise ValueError("The backward run left no feature; nothing to compare.")
    return StrategyOutcome(
        strategy=STRATEGY_BACKWARD,
        selected=result.remaining,
        evaluation=evaluate_feature_subset(
            dataset, plan, result.remaining, config=settings
        ),
        n_evaluations=result.n_evaluations,
        n_model_fits=result.n_model_fits,
        stopped_reason=result.stopped_reason,
        detail={
            "search_best_score": result.best_score,
            "removed": list(result.removed),
            "n_rounds": len(result.history),
        },
    )


def feature_stability(selections: Mapping[str, Sequence[str]]) -> dict:
    """Selection frequency per feature across the given selections.

    Call it with several selections (for example one per fold or one per seed).
    With a single selection the frequencies are trivially 0 or 1, and
    ``n_selections`` makes that basis explicit.
    """
    names = sorted({feature for subset in selections.values() for feature in subset})
    counter: Counter[str] = Counter()
    for subset in selections.values():
        counter.update(set(subset))
    total = len(selections)
    return {
        "n_selections": total,
        "frequency": {
            name: {
                "count": counter.get(name, 0),
                "fraction": (counter.get(name, 0) / total) if total else 0.0,
            }
            for name in names
        },
    }


@dataclass(frozen=True)
class ComparisonReport:
    """Side-by-side table plus the stability block, JSON-serialisable."""

    rows: tuple[dict, ...]
    stability: dict
    notes: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "rows": list(self.rows),
            "stability": self.stability,
            "notes": list(self.notes),
        }


def compare_outcomes(outcomes: Sequence[StrategyOutcome]) -> ComparisonReport:
    """Build the comparison table and an honest stability block.

    Per-fold stability is **not** produced: each strategy yields one selection
    for the whole plan, so there is no per-fold selection to compare yet. The
    report says so instead of inventing a number.
    """
    if not outcomes:
        raise ValueError("outcomes is empty.")
    names = [outcome.strategy for outcome in outcomes]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate strategy in outcomes: {names}.")

    selections = {outcome.strategy: outcome.selected for outcome in outcomes}
    return ComparisonReport(
        rows=tuple(outcome.as_dict() for outcome in outcomes),
        stability={
            "per_fold": {
                "available": False,
                "reason": (
                    "each strategy produces a single selection for the whole plan, "
                    "not one selection per fold"
                ),
            },
            "across_strategies": feature_stability(selections),
        },
        notes=(
            "Every strategy is scored on the same grouped folds, masks and estimator.",
            "Final subsets are re-scored with no variance filter, so all strategies "
            "go through one evaluation path.",
            "Subset sizes differ by construction; they are reported, never equalised.",
        ),
    )
