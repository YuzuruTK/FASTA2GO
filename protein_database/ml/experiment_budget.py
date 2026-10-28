"""Run budget: limits checked before each task, costs observed afterwards.

Two ideas are kept strictly apart:

* **estimate** (:class:`CostEstimate`) - what a task is expected to cost, derived
  from the plan's dimensions (folds, features, labels, selection parameters).
  It is an upper bound, flagged with ``exact=False`` when it cannot be exact.
* **observation** (:class:`BudgetLedger`) - what the runner actually charged,
  taken from the evaluation/selection objects that ran.

Five counters exist because they measure different things: experimental tasks,
fold evaluations, estimator fits, binary fits internal to One-vs-Rest, and
candidate evaluations of a feature search. Only the first four groups are
*limited*: ``binary_fits`` is recorded but not budgeted, because it is an
internal detail of ``OneVsRestClassifier`` and cannot be capped without
changing the estimator.

Time limit
----------
``max_seconds`` is a **cooperative** limit: it is checked before each task
starts. A task already running is never interrupted, because scikit-learn
estimators have no safe cancellation primitive and killing a thread or process
mid-fit would leave resources behind. No ``cancelled`` state is therefore ever
produced. Memory is not budgeted at all: no portable hard limit can be
guaranteed here, so none is promised.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

LIMIT_MAX_TASKS = "max_tasks"
LIMIT_MAX_FOLD_EVALUATIONS = "max_fold_evaluations"
LIMIT_MAX_ESTIMATOR_FITS = "max_estimator_fits"
LIMIT_MAX_SELECTION_CANDIDATE_EVALUATIONS = (
    "max_selection_candidate_evaluations"
)
LIMIT_MAX_SECONDS = "max_seconds"

LIMITS = (
    LIMIT_MAX_TASKS,
    LIMIT_MAX_FOLD_EVALUATIONS,
    LIMIT_MAX_ESTIMATOR_FITS,
    LIMIT_MAX_SELECTION_CANDIDATE_EVALUATIONS,
    LIMIT_MAX_SECONDS,
)


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of one task, with the reason it may be inexact."""

    tasks: int = 1
    fold_evaluations: int = 0
    estimator_fits: int = 0
    binary_fits: int = 0
    selection_candidate_evaluations: int = 0
    exact: bool = True
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "tasks": self.tasks,
            "fold_evaluations": self.fold_evaluations,
            "estimator_fits": self.estimator_fits,
            "binary_fits": self.binary_fits,
            "selection_candidate_evaluations": (
                self.selection_candidate_evaluations
            ),
            "exact": self.exact,
            "notes": list(self.notes),
        }

    def plus(self, other: "CostEstimate") -> "CostEstimate":
        notes: list[str] = list(self.notes)
        for note in other.notes:
            if note not in notes:
                notes.append(note)
        return CostEstimate(
            tasks=self.tasks + other.tasks,
            fold_evaluations=self.fold_evaluations + other.fold_evaluations,
            estimator_fits=self.estimator_fits + other.estimator_fits,
            binary_fits=self.binary_fits + other.binary_fits,
            selection_candidate_evaluations=(
                self.selection_candidate_evaluations
                + other.selection_candidate_evaluations
            ),
            exact=self.exact and other.exact,
            notes=tuple(notes),
        )


def total_cost(estimates: object) -> CostEstimate:
    """Sum an iterable of estimates (an empty iterable gives a zero estimate)."""
    total = CostEstimate(tasks=0)
    for estimate in estimates:  # type: ignore[union-attr]
        total = total.plus(estimate)
    return total


@dataclass(frozen=True)
class RunBudget:
    """Hard limits, all optional; ``None`` means unlimited for that counter."""

    max_tasks: int | None = None
    max_fold_evaluations: int | None = None
    max_estimator_fits: int | None = None
    max_selection_candidate_evaluations: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in LIMITS[:-1]:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"{name} must be None or an integer >= 1; got {value!r}."
                )
        if self.max_seconds is not None:
            if not isinstance(self.max_seconds, (int, float)) or isinstance(
                self.max_seconds, bool
            ):
                raise ValueError(
                    f"max_seconds must be None or a number; got {self.max_seconds!r}."
                )
            if not math.isfinite(float(self.max_seconds)) or self.max_seconds <= 0:
                raise ValueError(
                    f"max_seconds must be finite and > 0; got {self.max_seconds!r}."
                )

    @property
    def unlimited(self) -> bool:
        return all(getattr(self, name) is None for name in LIMITS)

    def as_dict(self) -> dict:
        return {
            "max_tasks": self.max_tasks,
            "max_fold_evaluations": self.max_fold_evaluations,
            "max_estimator_fits": self.max_estimator_fits,
            "max_selection_candidate_evaluations": (
                self.max_selection_candidate_evaluations
            ),
            "max_seconds": (
                None if self.max_seconds is None else float(self.max_seconds)
            ),
            "time_limit_mode": "cooperative, checked before each task",
            "memory_limit": None,
            "notes": [
                "binary_fits are recorded but not budgeted",
                "a running estimator is never interrupted",
            ],
        }


@dataclass
class BudgetLedger:
    """Observed counters plus the limits they are compared against."""

    budget: RunBudget = field(default_factory=RunBudget)
    tasks_started: int = 0
    fold_evaluations: int = 0
    estimator_fits: int = 0
    binary_fits: int = 0
    selection_candidate_evaluations: int = 0
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def consumed(self) -> dict:
        return {
            "tasks_started": self.tasks_started,
            "fold_evaluations": self.fold_evaluations,
            "estimator_fits": self.estimator_fits,
            "binary_fits": self.binary_fits,
            "selection_candidate_evaluations": self.selection_candidate_evaluations,
            "elapsed_seconds": self.elapsed_seconds(),
        }

    def remaining(self) -> dict:
        elapsed = self.elapsed_seconds()
        return {
            LIMIT_MAX_TASKS: _remaining(self.budget.max_tasks, self.tasks_started),
            LIMIT_MAX_FOLD_EVALUATIONS: _remaining(
                self.budget.max_fold_evaluations, self.fold_evaluations
            ),
            LIMIT_MAX_ESTIMATOR_FITS: _remaining(
                self.budget.max_estimator_fits, self.estimator_fits
            ),
            LIMIT_MAX_SELECTION_CANDIDATE_EVALUATIONS: _remaining(
                self.budget.max_selection_candidate_evaluations,
                self.selection_candidate_evaluations,
            ),
            LIMIT_MAX_SECONDS: _remaining(self.budget.max_seconds, elapsed),
        }

    def can_afford(self, cost: CostEstimate) -> tuple[bool, str | None]:
        """Whether ``cost`` fits in the remaining budget, with the blocking reason."""
        checks = (
            (LIMIT_MAX_TASKS, self.budget.max_tasks, self.tasks_started, cost.tasks),
            (
                LIMIT_MAX_FOLD_EVALUATIONS,
                self.budget.max_fold_evaluations,
                self.fold_evaluations,
                cost.fold_evaluations,
            ),
            (
                LIMIT_MAX_ESTIMATOR_FITS,
                self.budget.max_estimator_fits,
                self.estimator_fits,
                cost.estimator_fits,
            ),
            (
                LIMIT_MAX_SELECTION_CANDIDATE_EVALUATIONS,
                self.budget.max_selection_candidate_evaluations,
                self.selection_candidate_evaluations,
                cost.selection_candidate_evaluations,
            ),
        )
        for name, limit, used, wanted in checks:
            if limit is None:
                continue
            if used + wanted > limit:
                return False, f"{name}: {used} used + {wanted} estimated > {limit}"
        if self.budget.max_seconds is not None:
            elapsed = self.elapsed_seconds()
            if elapsed >= float(self.budget.max_seconds):
                return (
                    False,
                    f"{LIMIT_MAX_SECONDS}: {elapsed:.3f}s elapsed >= "
                    f"{float(self.budget.max_seconds):.3f}s (cooperative limit)",
                )
        return True, None

    def charge(
        self,
        *,
        tasks: int = 1,
        fold_evaluations: int = 0,
        estimator_fits: int = 0,
        binary_fits: int = 0,
        selection_candidate_evaluations: int = 0,
    ) -> None:
        """Add observed costs; negative values are a programming error."""
        values = {
            "tasks": tasks,
            "fold_evaluations": fold_evaluations,
            "estimator_fits": estimator_fits,
            "binary_fits": binary_fits,
            "selection_candidate_evaluations": selection_candidate_evaluations,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0; got {value!r}.")
        self.tasks_started += tasks
        self.fold_evaluations += fold_evaluations
        self.estimator_fits += estimator_fits
        self.binary_fits += binary_fits
        self.selection_candidate_evaluations += selection_candidate_evaluations

    def charge_observed(self, observed: Mapping[str, int]) -> None:
        """Charge a mapping of observed counts (missing keys count as zero)."""
        self.charge(
            tasks=int(observed.get("tasks", 1)),
            fold_evaluations=int(observed.get("fold_evaluations", 0)),
            estimator_fits=int(observed.get("estimator_fits", 0)),
            binary_fits=int(observed.get("binary_fits", 0)),
            selection_candidate_evaluations=int(
                observed.get("selection_candidate_evaluations", 0)
            ),
        )

    def snapshot(self) -> dict:
        return {
            "limits": self.budget.as_dict(),
            "consumed": self.consumed(),
            "remaining": self.remaining(),
        }


def _remaining(limit: float | None, used: float) -> float | None:
    return None if limit is None else max(0.0, float(limit) - float(used))
