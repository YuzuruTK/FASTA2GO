"""Audit contract for forward-selection rounds (no model fitting in Etapa 1).

This module records **decisions**, not models. The candidate loop that produces
the scores arrives in a later step; here we fix the vocabulary and enforce the
rules from the plan:

* **R1** ``reference_score`` is the score of the currently selected set;
* **R2** the stop decision uses only the best candidate's gain;
* **R3** a rejected candidate never enters ``selected``;
* **R4** the first round has no reference and is always accepted;
* **R5** no ordering is imposed on candidate scores;
* **R6** accepted rounds are non-decreasing (a consequence of R2 and
  ``min_improvement >= 0``).

Enforcing these at record time turns an easy-to-miss methodological bug into an
immediate ``ValueError``, and keeps the audit trail (order, gain, ties, cost)
available for ``selection_history.csv``.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from protein_database.ml.dataset import ProteinDataset, validate_dataset
from protein_database.ml.evaluation import (
    evaluable_label_mask,
    fit_label_mask,
    multilabel_metrics,
)
from protein_database.ml.models import LogisticConfig, build_estimator, describe_estimator
from protein_database.ml.prefilter import fit_variance_filter
from protein_database.ml.splits import Fold, SplitPlan, split_coverage

#: Column order of ``selection_history.csv`` (stable, auditable artifact).
HISTORY_COLUMNS = (
    "round",
    "candidate",
    "score",
    "reference_score",
    "gain",
    "accepted",
    "tied_candidates",
    "n_candidates_evaluated",
    "n_model_fits",
    "converged",
    "elapsed_seconds",
    "stop_reason",
)

STOP_MAX_FEATURES = "max_features"
STOP_MIN_IMPROVEMENT = "min_improvement"
STOP_NO_CANDIDATES = "no_candidates"
STOP_BUDGET = "budget"
STOP_RUNTIME = "runtime"
STOP_MEMORY = "memory"
STOP_COMPLETED = "completed"
STOP_NO_EVALUABLE_LABELS = "no_evaluable_labels"

#: Column order of ``selection_candidates.csv`` (every candidate of every round).
CANDIDATE_COLUMNS = ("round", "feature", "score", "score_std")


@dataclass(frozen=True)
class FoldScore:
    """Metrics of one inner fold for the candidate that won its round."""

    fold: str
    macro_f1: float
    n_labels_evaluated: int
    n_labels_total: int
    n_fit_samples: int
    n_validation_samples: int
    n_model_fits: int
    converged: bool = True
    excluded_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    """Score of one candidate in one round (mean over folds)."""

    feature: str
    score: float
    score_std: float = 0.0


@dataclass(frozen=True)
class SelectionRound:
    """One evaluated round: the best candidate and what was decided about it."""

    round: int
    candidate: str
    score: float
    reference_score: float | None
    gain: float | None
    accepted: bool
    tied_candidates: tuple[str, ...] = ()
    n_candidates_evaluated: int = 0
    n_model_fits: int = 0
    converged: bool = True
    elapsed_seconds: float = 0.0
    stop_reason: str | None = None
    score_std: float = 0.0
    fold_scores: tuple[FoldScore, ...] = ()
    candidates: tuple[CandidateScore, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    """Final outcome of a forward-selection run, with its audit trail."""

    selected: tuple[str, ...]
    history: tuple[SelectionRound, ...]
    stopped_reason: str
    n_evaluations: int
    n_model_fits: int
    best_score: float | None
    scoring: str = "macro_f1"

    def as_rows(self) -> list[dict]:
        """Rows ready for ``selection_history.csv`` (keys follow ``HISTORY_COLUMNS``)."""
        return [
            {
                "round": round_.round,
                "candidate": round_.candidate,
                "score": round_.score,
                "reference_score": round_.reference_score,
                "gain": round_.gain,
                "accepted": round_.accepted,
                "tied_candidates": ";".join(round_.tied_candidates),
                "n_candidates_evaluated": round_.n_candidates_evaluated,
                "n_model_fits": round_.n_model_fits,
                "converged": round_.converged,
                "elapsed_seconds": round_.elapsed_seconds,
                "stop_reason": round_.stop_reason,
            }
            for round_ in self.history
        ]

    def candidate_rows(self) -> list[dict]:
        """Rows ready for ``selection_candidates.csv`` (every candidate, every round)."""
        return [
            {
                "round": round_.round,
                "feature": candidate.feature,
                "score": candidate.score,
                "score_std": candidate.score_std,
            }
            for round_ in self.history
            for candidate in round_.candidates
        ]


class SelectionRecorder:
    """Records forward-selection rounds and enforces rules R1-R6.

    Args:
        min_improvement: Minimum gain in the selection metric required to accept
            a candidate. ``0`` accepts ties; negative values are rejected.
        scoring: Name of the selection metric, stored for the manifest.
    """

    def __init__(
        self, *, min_improvement: float = 1e-4, scoring: str = "macro_f1"
    ) -> None:
        if not isinstance(min_improvement, (int, float)) or not math.isfinite(
            min_improvement
        ):
            raise ValueError(
                f"min_improvement must be a finite number; got {min_improvement!r}."
            )
        if min_improvement < 0:
            raise ValueError(
                "min_improvement must be >= 0; a negative value would accept worse "
                f"candidates. Got {min_improvement!r}."
            )
        if not isinstance(scoring, str) or not scoring:
            raise ValueError(f"scoring must be a non-empty string; got {scoring!r}.")
        self._min_improvement = float(min_improvement)
        self._scoring = scoring
        self._rounds: list[SelectionRound] = []
        self._selected: list[str] = []
        self._stop_reason: str | None = None

    @property
    def selected(self) -> tuple[str, ...]:
        """Features accepted so far, in selection order."""
        return tuple(self._selected)

    @property
    def reference_score(self) -> float | None:
        """Score of the currently selected set (``None`` before the first round)."""
        for round_ in reversed(self._rounds):
            if round_.accepted:
                return round_.score
        return None

    @property
    def stop_reason(self) -> str | None:
        """Reason recorded by the last :meth:`stop` or rejected round, if any."""
        return self._stop_reason

    def record(
        self,
        candidate: str,
        score: float,
        *,
        tied_candidates: tuple[str, ...] | list[str] = (),
        n_candidates_evaluated: int,
        n_model_fits: int = 0,
        converged: bool = True,
        elapsed_seconds: float = 0.0,
        score_std: float = 0.0,
        fold_scores: tuple[FoldScore, ...] | list[FoldScore] = (),
        candidates: tuple[CandidateScore, ...] | list[CandidateScore] = (),
    ) -> SelectionRound:
        """Record one round and decide whether ``candidate`` is accepted.

        Raises:
            RuntimeError: When a round is recorded after the run has stopped.
            ValueError: On an empty/duplicate candidate, a non-finite score, a
                non-positive candidate count, a negative fit count, or a tie
                list that does not contain the winning candidate.
        """
        if self._stop_reason is not None:
            raise RuntimeError(
                f"cannot record a round after stop reason {self._stop_reason!r}."
            )
        if not isinstance(candidate, str) or not candidate:
            raise ValueError(f"candidate must be a non-empty string; got {candidate!r}.")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"score must be finite; got {score!r}.")
        spread = float(score_std)
        if not math.isfinite(spread) or spread < 0:
            raise ValueError(f"score_std must be finite and >= 0; got {score_std!r}.")
        if candidate in self._selected:
            raise ValueError(f"candidate {candidate!r} was already selected.")
        if not isinstance(n_candidates_evaluated, int) or n_candidates_evaluated < 1:
            raise ValueError(
                "n_candidates_evaluated must be an integer >= 1; got "
                f"{n_candidates_evaluated!r}."
            )
        if not isinstance(n_model_fits, int) or n_model_fits < 0:
            raise ValueError(
                f"n_model_fits must be an integer >= 0; got {n_model_fits!r}."
            )
        ties = tuple(tied_candidates)
        if ties and candidate not in ties:
            raise ValueError(
                f"the winning candidate {candidate!r} must be one of the tied "
                f"candidates {ties!r}."
            )

        reference = self.reference_score
        gain = None if reference is None else value - reference
        accepted = gain is None or gain >= self._min_improvement

        round_ = SelectionRound(
            round=len(self._rounds) + 1,
            candidate=candidate,
            score=value,
            reference_score=reference,
            gain=gain,
            accepted=accepted,
            tied_candidates=ties,
            n_candidates_evaluated=int(n_candidates_evaluated),
            n_model_fits=int(n_model_fits),
            converged=bool(converged),
            elapsed_seconds=float(elapsed_seconds),
            stop_reason=None if accepted else STOP_MIN_IMPROVEMENT,
            score_std=spread,
            fold_scores=tuple(fold_scores),
            candidates=tuple(candidates),
        )
        self._rounds.append(round_)
        if accepted:
            self._selected.append(candidate)
        else:
            self._stop_reason = STOP_MIN_IMPROVEMENT
        return round_

    def stop(self, reason: str) -> None:
        """Close the run without recording a round (budget, runtime, memory...).

        Raises:
            RuntimeError: When the run already stopped.
            ValueError: On an empty reason.
        """
        if self._stop_reason is not None:
            raise RuntimeError(
                f"the run already stopped with reason {self._stop_reason!r}."
            )
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"reason must be a non-empty string; got {reason!r}.")
        self._stop_reason = reason

    def finalize(self, *, stopped_reason: str | None = None) -> SelectionResult:
        """Build the immutable result with its audit trail.

        Args:
            stopped_reason: Overrides the inferred reason (for example
                ``"max_features"`` when the cap was reached). Contradicting an
                already recorded stop raises.

        Raises:
            ValueError: When ``stopped_reason`` contradicts a recorded stop.
        """
        if (
            stopped_reason is not None
            and self._stop_reason is not None
            and stopped_reason != self._stop_reason
        ):
            raise ValueError(
                f"stopped_reason {stopped_reason!r} contradicts the recorded stop "
                f"reason {self._stop_reason!r}."
            )
        if stopped_reason is not None:
            reason = stopped_reason
        elif self._stop_reason is not None:
            reason = self._stop_reason
        elif self._rounds:
            reason = STOP_COMPLETED
        else:
            reason = STOP_NO_CANDIDATES

        return SelectionResult(
            selected=tuple(self._selected),
            history=tuple(self._rounds),
            stopped_reason=reason,
            n_evaluations=sum(r.n_candidates_evaluated for r in self._rounds),
            n_model_fits=sum(r.n_model_fits for r in self._rounds),
            best_score=self.reference_score,
            scoring=self._scoring,
        )


# ---------------------------------------------------------------------------
# Forward selection over grouped inner folds
# ---------------------------------------------------------------------------

#: Absolute tolerance used to declare two candidate scores tied.
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ForwardSelectionConfig:
    """Parameters of one forward-selection run.

    ``min_improvement`` follows the same contract as :class:`SelectionRecorder`:
    a candidate is accepted only when its score exceeds the score of the current
    selected set by at least this margin. ``max_features`` is a ceiling, not a
    target.

    ``max_evaluations`` caps candidate evaluations (candidate x fold) and exists
    because forward selection is quadratic: with ``p`` candidates and ``k``
    accepted features the run performs ``sum(p - i)`` evaluations, each of which
    fits one model per fold.
    """

    max_features: int = 50
    min_improvement: float = 1e-4
    max_evaluations: int | None = None
    variance_threshold: float = 0.0
    min_positives_fit: int = 2
    min_negatives_fit: int = 2
    min_positives_eval: int = 1
    estimator: LogisticConfig = field(default_factory=LogisticConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.max_features, int) or self.max_features < 1:
            raise ValueError(
                f"max_features must be an integer >= 1; got {self.max_features!r}."
            )
        if not isinstance(self.min_improvement, (int, float)) or not math.isfinite(
            self.min_improvement
        ):
            raise ValueError(
                "min_improvement must be a finite number; got "
                f"{self.min_improvement!r}."
            )
        if self.min_improvement < 0:
            raise ValueError(
                f"min_improvement must be >= 0; got {self.min_improvement!r}."
            )
        if self.max_evaluations is not None and (
            not isinstance(self.max_evaluations, int) or self.max_evaluations < 1
        ):
            raise ValueError(
                "max_evaluations must be None or an integer >= 1; got "
                f"{self.max_evaluations!r}."
            )
        if (
            not isinstance(self.variance_threshold, (int, float))
            or not math.isfinite(self.variance_threshold)
            or self.variance_threshold < 0
        ):
            raise ValueError(
                f"variance_threshold must be >= 0; got {self.variance_threshold!r}."
            )


@dataclass(frozen=True)
class _FoldContext:
    """Everything one fold needs to score a candidate, fitted on its train block."""

    name: str
    X_train: np.ndarray
    X_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    labels_fit: np.ndarray
    labels_eval: np.ndarray
    eval_positions: np.ndarray
    columns: dict[str, int]
    available: frozenset[str]
    excluded_labels: tuple[str, ...]
    n_labels_total: int


def _build_contexts(
    dataset: ProteinDataset,
    plan: SplitPlan,
    config: ForwardSelectionConfig,
    *,
    apply_variance_filter: bool = True,
) -> tuple[list[_FoldContext], tuple[str, ...]]:
    """Fit the variance filter and the label masks of every fold.

    Only ``fold.train`` is used to fit the variance filter; the validation block
    is transformed with the filter learned on the training rows. Folds without a
    single evaluable label are returned separately instead of producing NaN.

    ``apply_variance_filter=False`` keeps every feature, which is what the
    "all features, no selection" baseline needs; the estimator's scaler is
    already robust to a constant column.
    """
    contexts: list[_FoldContext] = []
    skipped: list[str] = []
    for fold in plan.folds:
        y_train = dataset.y[fold.train]
        y_validation = dataset.y[fold.validation]
        fit_mask = fit_label_mask(
            y_train,
            min_positives=config.min_positives_fit,
            min_negatives=config.min_negatives_fit,
        )
        eval_mask = evaluable_label_mask(
            y_train,
            y_validation,
            min_positives_fit=config.min_positives_fit,
            min_negatives_fit=config.min_negatives_fit,
            min_positives_eval=config.min_positives_eval,
        )
        labels_fit = np.flatnonzero(fit_mask)
        labels_eval = np.flatnonzero(eval_mask)
        if labels_eval.size == 0 or labels_fit.size == 0:
            skipped.append(fold.name)
            continue

        if apply_variance_filter:
            variance = fit_variance_filter(
                dataset.X[fold.train],
                dataset.feature_names,
                threshold=config.variance_threshold,
            )
            X_train = variance.transform(dataset.X[fold.train])
            X_validation = variance.transform(dataset.X[fold.validation])
            kept = variance.kept
        else:
            X_train = dataset.X[fold.train]
            X_validation = dataset.X[fold.validation]
            kept = dataset.feature_names

        contexts.append(
            _FoldContext(
                name=fold.name,
                X_train=X_train,
                X_validation=X_validation,
                y_train=y_train,
                y_validation=y_validation,
                labels_fit=labels_fit,
                labels_eval=labels_eval,
                eval_positions=np.searchsorted(labels_fit, labels_eval),
                columns={name: index for index, name in enumerate(kept)},
                available=frozenset(kept),
                excluded_labels=tuple(
                    name
                    for index, name in enumerate(dataset.label_names)
                    if not eval_mask[index]
                ),
                n_labels_total=len(dataset.label_names),
            )
        )
    return contexts, tuple(skipped)


def _score_names(
    names: tuple[str, ...],
    contexts: list[_FoldContext],
    config: ForwardSelectionConfig,
    *,
    model: object | None = None,
) -> tuple[float, float, tuple[FoldScore, ...], int, bool]:
    """Score a fixed feature subset on every fold; return mean, std and detail.

    Each fold fits a fresh estimator on its own training block, so nothing is
    shared between folds and no validation row ever reaches a ``fit``.

    ``model`` overrides ``config.estimator``; the default keeps the selection
    behaviour bit-identical to the previous implementation.
    """
    scores: list[float] = []
    fold_scores: list[FoldScore] = []
    total_fits = 0
    converged = True

    for context in contexts:
        columns = [context.columns[name] for name in names]
        estimator = build_estimator(model or config.estimator)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(
                context.X_train[:, columns], context.y_train[:, context.labels_fit]
            )
            predictions = estimator.predict(context.X_validation[:, columns])
        predictions = predictions[:, context.eval_positions]
        metrics = multilabel_metrics(
            context.y_validation[:, context.labels_eval], predictions
        )
        fold_converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        converged = converged and fold_converged
        total_fits += int(context.labels_fit.size)
        scores.append(float(metrics["macro_f1"]))
        fold_scores.append(
            FoldScore(
                fold=context.name,
                macro_f1=float(metrics["macro_f1"]),
                n_labels_evaluated=int(context.labels_eval.size),
                n_labels_total=context.n_labels_total,
                n_fit_samples=int(context.X_train.shape[0]),
                n_validation_samples=int(context.X_validation.shape[0]),
                n_model_fits=int(context.labels_fit.size),
                converged=fold_converged,
                excluded_labels=context.excluded_labels,
            )
        )

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        tuple(fold_scores),
        total_fits,
        converged,
    )


def _evaluate_candidate(
    selected: tuple[str, ...],
    candidate: str,
    contexts: list[_FoldContext],
    config: ForwardSelectionConfig,
) -> tuple[float, float, tuple[FoldScore, ...], int, bool]:
    """Score ``selected + candidate``, the forward-selection candidate rule."""
    return _score_names((*selected, candidate), contexts, config)


@dataclass(frozen=True)
class ForwardSelectionResult:
    """Selection audit plus the pre-filter and coverage context of the run."""

    selection: SelectionResult
    feature_names: tuple[str, ...]
    prefilter_kept: tuple[str, ...]
    prefilter_dropped: tuple[str, ...]
    excluded_folds: tuple[str, ...]
    fold_label_coverage: tuple[dict, ...]
    config: ForwardSelectionConfig

    def report(self) -> dict:
        """Audit payload for the experiment manifest (JSON-serialisable)."""
        return {
            "config": {
                "max_features": self.config.max_features,
                "min_improvement": self.config.min_improvement,
                "max_evaluations": self.config.max_evaluations,
                "variance_threshold": self.config.variance_threshold,
                "min_positives_fit": self.config.min_positives_fit,
                "min_negatives_fit": self.config.min_negatives_fit,
                "min_positives_eval": self.config.min_positives_eval,
                "estimator": describe_estimator(self.config.estimator),
            },
            "features_in": list(self.feature_names),
            "prefilter_kept": list(self.prefilter_kept),
            "prefilter_dropped": list(self.prefilter_dropped),
            "excluded_folds": list(self.excluded_folds),
            "selected": list(self.selection.selected),
            "stopped_reason": self.selection.stopped_reason,
            "best_score": self.selection.best_score,
            "n_evaluations": self.selection.n_evaluations,
            "n_model_fits": self.selection.n_model_fits,
            "history": self.selection.as_rows(),
            "candidates": self.selection.candidate_rows(),
            "fold_label_coverage": list(self.fold_label_coverage),
        }


def _validate_plan_indices(plan: SplitPlan, n_samples: int) -> None:
    blocks = [("test", plan.test)]
    for fold in plan.folds:
        blocks.append((f"{fold.name}/train", fold.train))
        blocks.append((f"{fold.name}/validation", fold.validation))
    for name, indices in blocks:
        if indices.size == 0:
            raise ValueError(f"{name} is empty.")
        if indices.min() < 0 or indices.max() >= n_samples:
            raise ValueError(
                f"{name} has indices outside [0, {n_samples}); got "
                f"[{int(indices.min())}, {int(indices.max())}]."
            )


def run_forward_selection(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> ForwardSelectionResult:
    """Run forward selection over the inner folds of ``plan``.

    Contract, in order of precedence:

    1. the candidate set is the features kept by the variance filter of **every**
       fold (a feature constant in any fold's training block is dropped and
       reported, never silently re-added);
    2. a candidate is scored by the mean macro-F1 over folds, computed on the
       evaluable-label mask of each fold; the spread across folds is recorded;
    3. the winner is the highest mean; ties are broken by the reference feature
       order (first wins) and recorded in ``tied_candidates``;
    4. the run stops on ``min_improvement`` (decided by the recorder), on
       ``max_features``, on ``max_evaluations`` or when candidates run out, and
       never records a round after stopping;
    5. the external test block is never read: only ``fold.train`` and
       ``fold.validation`` are used.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: On invalid plan indices or an invalid configuration.
    """
    settings = config or ForwardSelectionConfig()
    validate_dataset(dataset)
    _validate_plan_indices(plan, dataset.n_samples)

    recorder = SelectionRecorder(min_improvement=settings.min_improvement)
    contexts, excluded_folds = _build_contexts(dataset, plan, settings)

    if not contexts:
        recorder.stop(STOP_NO_EVALUABLE_LABELS)
        return ForwardSelectionResult(
            selection=recorder.finalize(),
            feature_names=dataset.feature_names,
            prefilter_kept=(),
            prefilter_dropped=(),
            excluded_folds=excluded_folds,
            fold_label_coverage=(),
            config=settings,
        )

    available = tuple(
        name
        for name in dataset.feature_names
        if all(name in context.available for context in contexts)
    )
    dropped = tuple(
        name for name in dataset.feature_names if name not in set(available)
    )

    remaining = list(available)
    evaluations = 0
    if not remaining:
        recorder.stop(STOP_NO_CANDIDATES)

    while remaining and len(recorder.selected) < settings.max_features:
        started = time.perf_counter()
        round_candidates: list[CandidateScore] = []
        details: dict[str, tuple[tuple[FoldScore, ...], int, bool]] = {}
        aborted = False

        for name in remaining:
            if (
                settings.max_evaluations is not None
                and evaluations >= settings.max_evaluations
            ):
                aborted = True
                break
            score, spread, fold_scores, fits, converged = _evaluate_candidate(
                recorder.selected, name, contexts, settings
            )
            evaluations += len(contexts)
            round_candidates.append(CandidateScore(name, score, spread))
            details[name] = (fold_scores, fits, converged)

        if aborted:
            recorder.stop(STOP_BUDGET)
            break
        if not round_candidates:
            break

        best = max(candidate.score for candidate in round_candidates)
        tied = tuple(
            candidate.feature
            for candidate in round_candidates
            if abs(candidate.score - best) <= TIE_TOLERANCE
        )
        winner = tied[0]
        winner_folds, winner_fits, winner_converged = details[winner]
        winner_spread = next(
            candidate.score_std
            for candidate in round_candidates
            if candidate.feature == winner
        )

        round_ = recorder.record(
            winner,
            best,
            tied_candidates=tied,
            n_candidates_evaluated=len(round_candidates),
            n_model_fits=winner_fits,
            converged=winner_converged,
            elapsed_seconds=time.perf_counter() - started,
            score_std=winner_spread,
            fold_scores=winner_folds,
            candidates=tuple(round_candidates),
        )
        remaining.remove(winner)
        if not round_.accepted:
            break

    if recorder.stop_reason is not None:
        selection = recorder.finalize()
    else:
        reached_cap = len(recorder.selected) >= settings.max_features
        selection = recorder.finalize(
            stopped_reason=STOP_MAX_FEATURES if reached_cap else STOP_NO_CANDIDATES
        )

    return ForwardSelectionResult(
        selection=selection,
        feature_names=dataset.feature_names,
        prefilter_kept=available,
        prefilter_dropped=dropped,
        excluded_folds=excluded_folds,
        fold_label_coverage=tuple(
            split_coverage(
                dataset.y,
                plan,
                dataset.label_names,
                min_positives=settings.min_positives_fit,
                include_test=False,
            )
        ),
        config=settings,
    )


# ---------------------------------------------------------------------------
# Shared subset evaluation and backward elimination
# ---------------------------------------------------------------------------

STOP_MIN_FEATURES = "min_features"


def eligible_features(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
) -> tuple[str, ...]:
    """Features kept by the variance filter of **every** inner fold.

    Lives outside the strategies so baselines and both search directions use one
    definition of "eligible". The test block is not touched: only ``fold.train``
    enters the variance fit.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: On invalid plan indices.
    """
    settings = config or ForwardSelectionConfig()
    validate_dataset(dataset)
    _validate_plan_indices(plan, dataset.n_samples)
    kept: set[str] | None = None
    for fold in plan.folds:
        variance = fit_variance_filter(
            dataset.X[fold.train],
            dataset.feature_names,
            threshold=settings.variance_threshold,
        )
        fold_kept = set(variance.kept)
        kept = fold_kept if kept is None else kept & fold_kept
    universe = kept or set()
    return tuple(name for name in dataset.feature_names if name in universe)


@dataclass(frozen=True)
class SubsetEvaluation:
    """Mean and per-fold macro-F1 of one fixed feature subset."""

    features: tuple[str, ...]
    fold_scores: tuple[FoldScore, ...]
    mean_macro_f1: float
    std_macro_f1: float
    n_model_fits: int
    converged: bool
    excluded_folds: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "features": list(self.features),
            "n_features": len(self.features),
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "n_model_fits": self.n_model_fits,
            "converged": self.converged,
            "excluded_folds": list(self.excluded_folds),
            "fold_scores": [
                {
                    "fold": fold.fold,
                    "macro_f1": fold.macro_f1,
                    "n_labels_evaluated": fold.n_labels_evaluated,
                    "n_validation_samples": fold.n_validation_samples,
                }
                for fold in self.fold_scores
            ],
        }


def evaluate_feature_subset(
    dataset: ProteinDataset,
    plan: SplitPlan,
    features: Sequence[str],
    *,
    config: ForwardSelectionConfig | None = None,
    model: object | None = None,
) -> SubsetEvaluation:
    """Evaluate one fixed subset with the same folds, masks and estimator.

    The subset is evaluated **without** any variance filtering, so "all
    features" really means all of them; the filtered baseline passes the
    intersection from :func:`eligible_features` instead.

    ``model`` selects the estimator: ``None`` keeps ``config.estimator`` (the
    frozen logistic reference), so callers of the selection code are unaffected
    while the model comparison can evaluate other families on the same folds.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: If the subset is empty, has unknown features, or no fold has
            an evaluable label.
    """
    settings = config or ForwardSelectionConfig()
    validate_dataset(dataset)
    _validate_plan_indices(plan, dataset.n_samples)

    names = tuple(features)
    if not names:
        raise ValueError("features is empty.")
    unknown = [name for name in names if name not in set(dataset.feature_names)]
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown[:5]}.")

    contexts, excluded = _build_contexts(
        dataset, plan, settings, apply_variance_filter=False
    )
    if not contexts:
        raise ValueError(
            "No fold has an evaluable label; the subset cannot be scored."
        )
    score, spread, fold_scores, fits, converged = _score_names(
        names, contexts, settings, model=model
    )
    return SubsetEvaluation(
        features=names,
        fold_scores=fold_scores,
        mean_macro_f1=score,
        std_macro_f1=spread,
        n_model_fits=fits,
        converged=converged,
        excluded_folds=excluded,
    )


@dataclass(frozen=True)
class EliminationRound:
    """One backward round: the feature whose removal scored best, and the verdict."""

    round: int
    removed: str
    score: float
    reference_score: float | None
    gain: float | None
    accepted: bool
    tied_candidates: tuple[str, ...] = ()
    n_candidates_evaluated: int = 0
    n_model_fits: int = 0
    converged: bool = True
    elapsed_seconds: float = 0.0
    stop_reason: str | None = None
    score_std: float = 0.0
    fold_scores: tuple[FoldScore, ...] = ()
    candidates: tuple[CandidateScore, ...] = ()


@dataclass(frozen=True)
class BackwardEliminationResult:
    """Final subset, removal order and the audit trail of a backward run."""

    remaining: tuple[str, ...]
    removed: tuple[str, ...]
    history: tuple[EliminationRound, ...]
    stopped_reason: str
    n_evaluations: int
    n_model_fits: int
    best_score: float | None
    excluded_folds: tuple[str, ...]
    fold_label_coverage: tuple[dict, ...]
    config: ForwardSelectionConfig

    def candidate_rows(self) -> list[dict]:
        """Rows for ``selection_candidates.csv`` (every removal tried, every round)."""
        return [
            {
                "round": round_.round,
                "feature": candidate.feature,
                "score": candidate.score,
                "score_std": candidate.score_std,
            }
            for round_ in self.history
            for candidate in round_.candidates
        ]

    def report(self) -> dict:
        """Audit payload for the experiment manifest (JSON-serialisable)."""
        return {
            "config": {
                "min_features": len(self.remaining),
                "min_improvement": self.config.min_improvement,
                "max_evaluations": self.config.max_evaluations,
                "variance_threshold": self.config.variance_threshold,
                "min_positives_fit": self.config.min_positives_fit,
                "min_negatives_fit": self.config.min_negatives_fit,
                "min_positives_eval": self.config.min_positives_eval,
                "estimator": describe_estimator(self.config.estimator),
            },
            "remaining": list(self.remaining),
            "removed": list(self.removed),
            "excluded_folds": list(self.excluded_folds),
            "stopped_reason": self.stopped_reason,
            "best_score": self.best_score,
            "n_evaluations": self.n_evaluations,
            "n_model_fits": self.n_model_fits,
            "history": [
                {
                    "round": round_.round,
                    "removed": round_.removed,
                    "score": round_.score,
                    "reference_score": round_.reference_score,
                    "gain": round_.gain,
                    "accepted": round_.accepted,
                    "score_std": round_.score_std,
                    "tied_candidates": list(round_.tied_candidates),
                    "n_candidates_evaluated": round_.n_candidates_evaluated,
                    "n_model_fits": round_.n_model_fits,
                    "converged": round_.converged,
                    "elapsed_seconds": round_.elapsed_seconds,
                    "stop_reason": round_.stop_reason,
                    "fold_scores": [
                        {
                            "fold": fold.fold,
                            "macro_f1": fold.macro_f1,
                            "n_labels_evaluated": fold.n_labels_evaluated,
                            "n_labels_total": fold.n_labels_total,
                        }
                        for fold in round_.fold_scores
                    ],
                }
                for round_ in self.history
            ],
            "candidates": self.candidate_rows(),
            "fold_label_coverage": list(self.fold_label_coverage),
        }


def run_backward_elimination(
    dataset: ProteinDataset,
    plan: SplitPlan,
    *,
    config: ForwardSelectionConfig | None = None,
    min_features: int = 1,
) -> BackwardEliminationResult:
    """Remove one feature per round while the removal improves the score.

    Starting point: the features kept by the variance filter of **every** fold
    (the same eligibility rule as forward selection). Each round scores the
    current subset minus each remaining feature, on the same folds, masks,
    estimator and mean-aggregation as forward selection. The removal with the
    highest mean wins; ties go to the first feature in reference order.

    A removal is accepted when the gain over the current subset is at least
    ``min_improvement``; otherwise the run stops and the last round is recorded
    as rejected. Because ``min_improvement > 0``, a removal that leaves the
    score unchanged is **not** accepted - the conservative reading of the same
    contract used by forward selection.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: On invalid plan indices, ``min_features < 1`` or a config
            problem.
    """
    settings = config or ForwardSelectionConfig()
    validate_dataset(dataset)
    _validate_plan_indices(plan, dataset.n_samples)
    if not isinstance(min_features, int) or min_features < 1:
        raise ValueError(
            f"min_features must be an integer >= 1; got {min_features!r}."
        )

    contexts, excluded_folds = _build_contexts(dataset, plan, settings)
    if not contexts:
        return BackwardEliminationResult(
            remaining=(),
            removed=(),
            history=(),
            stopped_reason=STOP_NO_EVALUABLE_LABELS,
            n_evaluations=0,
            n_model_fits=0,
            best_score=None,
            excluded_folds=excluded_folds,
            fold_label_coverage=(),
            config=settings,
        )

    available = tuple(
        name
        for name in dataset.feature_names
        if all(name in context.available for context in contexts)
    )
    remaining = list(available)
    removed: list[str] = []
    history: list[EliminationRound] = []
    evaluations = 0
    total_fits = 0
    reference: float | None = None
    stopped: str | None = None

    if len(remaining) > min_features:
        reference, _, _, start_fits, _ = _score_names(
            tuple(remaining), contexts, settings
        )
        total_fits += start_fits
    else:
        stopped = STOP_MIN_FEATURES

    while stopped is None and len(remaining) > min_features:
        started = time.perf_counter()
        round_candidates: list[CandidateScore] = []
        details: dict[str, tuple[tuple[FoldScore, ...], int, bool]] = {}
        aborted = False

        for name in remaining:
            if (
                settings.max_evaluations is not None
                and evaluations >= settings.max_evaluations
            ):
                aborted = True
                break
            subset = tuple(feature for feature in remaining if feature != name)
            score, spread, fold_scores, fits, converged = _score_names(
                subset, contexts, settings
            )
            evaluations += len(contexts)
            round_candidates.append(CandidateScore(name, score, spread))
            details[name] = (fold_scores, fits, converged)

        if aborted:
            stopped = STOP_BUDGET
            break
        if not round_candidates:
            stopped = STOP_NO_CANDIDATES
            break

        best = max(candidate.score for candidate in round_candidates)
        tied = tuple(
            candidate.feature
            for candidate in round_candidates
            if abs(candidate.score - best) <= TIE_TOLERANCE
        )
        winner = tied[0]
        winner_folds, winner_fits, winner_converged = details[winner]
        winner_spread = next(
            candidate.score_std
            for candidate in round_candidates
            if candidate.feature == winner
        )
        gain = None if reference is None else best - reference
        accepted = gain is None or gain >= settings.min_improvement

        round_ = EliminationRound(
            round=len(history) + 1,
            removed=winner,
            score=best,
            reference_score=reference,
            gain=gain,
            accepted=accepted,
            tied_candidates=tied,
            n_candidates_evaluated=len(round_candidates),
            n_model_fits=winner_fits,
            converged=winner_converged,
            elapsed_seconds=time.perf_counter() - started,
            stop_reason=None if accepted else STOP_MIN_IMPROVEMENT,
            score_std=winner_spread,
            fold_scores=winner_folds,
            candidates=tuple(round_candidates),
        )
        history.append(round_)
        total_fits += winner_fits

        if not accepted:
            stopped = STOP_MIN_IMPROVEMENT
            break

        remaining.remove(winner)
        removed.append(winner)
        reference = best
        if len(remaining) <= min_features:
            stopped = STOP_MIN_FEATURES

    if stopped is None:
        stopped = STOP_NO_CANDIDATES

    return BackwardEliminationResult(
        remaining=tuple(remaining),
        removed=tuple(removed),
        history=tuple(history),
        stopped_reason=stopped,
        n_evaluations=evaluations,
        n_model_fits=total_fits,
        best_score=reference,
        excluded_folds=excluded_folds,
        fold_label_coverage=tuple(
            split_coverage(
                dataset.y,
                plan,
                dataset.label_names,
                min_positives=settings.min_positives_fit,
                include_test=False,
            )
        ),
        config=settings,
    )
