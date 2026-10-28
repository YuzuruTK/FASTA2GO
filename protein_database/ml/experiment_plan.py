"""Execution plan: what would run, in which order, at what estimated cost.

Planning is pure. It reads dataset *metadata* (shapes, feature and label names,
accessions) and the folds' index arrays, never the feature or label matrices and
never ``plan.test``. No estimator is built, no model is fitted, nothing is
scored: :func:`build_execution_plan` only enumerates tasks.

Ordering
--------
1. selection tasks (forward, then backward, as declared in the configuration);
2. evaluation tasks, grouped by strategy in the declared strategy order and, inside
   a group, in the declared model order.

The order comes from the configuration tuples, never from set/dict iteration, so
two equal configurations give an identical plan. Identical plans on two machines
require the same library versions for the *numbers*; the ordering and the ids are
version independent.

States used here and by the runner:

``planned``, ``running`` (transient, never persisted), ``completed``, ``failed``,
``skipped_budget``, ``skipped_unavailable_dependency``,
``skipped_dependency_failure``. There is **no** ``cancelled`` state because no
safe cancellation is implemented.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from protein_database.ml.dataset import ProteinDataset, validate_dataset
from protein_database.ml.experiment_budget import CostEstimate, total_cost
from protein_database.ml.experiment_config import (
    PROTOCOL_VERSION,
    SELECTION_BACKWARD,
    SELECTION_FORWARD,
    ExperimentConfig,
    ModelSpec,
    SelectionSpec,
    digest,
    selection_parameters,
)
from protein_database.ml.splits import SplitPlan
from protein_database.ml.strategies import (
    STRATEGY_ALL,
    STRATEGY_BACKWARD,
    STRATEGY_FORWARD,
    STRATEGY_VARIANCE,
)

STATUS_PLANNED = "planned"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED_BUDGET = "skipped_budget"
STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY = "skipped_unavailable_dependency"
STATUS_SKIPPED_DEPENDENCY_FAILURE = "skipped_dependency_failure"

STATUSES = (
    STATUS_PLANNED,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED_BUDGET,
    STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
    STATUS_SKIPPED_DEPENDENCY_FAILURE,
)
SKIPPED_STATUSES = (
    STATUS_SKIPPED_BUDGET,
    STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
    STATUS_SKIPPED_DEPENDENCY_FAILURE,
)
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, *SKIPPED_STATUSES)

KIND_SELECTION = "selection"
KIND_EVALUATION = "evaluation"

SUBSET_ALL = "all_features"
SUBSET_VARIANCE = "variance_prefilter"
SUBSET_SELECTION = "selection"

MAX_EXACT_BINARY_FITS_NOTE = (
    "binary_fits is an upper bound (folds x labels): per-fold label masks may "
    "exclude labels, which only lowers the real count"
)
SEARCH_UPPER_BOUND_NOTE = (
    "selection cost is an upper bound: the search can stop early on "
    "min_improvement, on max_features or when candidates run out"
)


def default_dependency_probe(module: str) -> bool:
    """Whether an optional dependency can be imported, without importing it."""
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True)
class SubsetDefinition:
    """Where the feature subset of a task comes from.

    ``kind="selection"`` refers to a selection task by id; the features
    themselves only exist after that task ran, so the plan never guesses them.
    """

    kind: str
    selection_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in (SUBSET_ALL, SUBSET_VARIANCE, SUBSET_SELECTION):
            raise ValueError(f"Unknown subset kind {self.kind!r}.")
        if self.kind == SUBSET_SELECTION and not self.selection_id:
            raise ValueError("A selection subset needs a selection_id.")
        if self.kind != SUBSET_SELECTION and self.selection_id:
            raise ValueError(f"Subset kind {self.kind!r} cannot have a selection_id.")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "selection_id": self.selection_id}


def estimate_evaluation_cost(*, n_folds: int, n_labels: int) -> CostEstimate:
    """One evaluation task: one pipeline fit per fold, labels x folds binaries."""
    if n_folds < 1 or n_labels < 1:
        raise ValueError("n_folds and n_labels must be >= 1.")
    return CostEstimate(
        tasks=1,
        fold_evaluations=n_folds,
        estimator_fits=n_folds,
        binary_fits=n_folds * n_labels,
        selection_candidate_evaluations=0,
        exact=False,
        notes=(MAX_EXACT_BINARY_FITS_NOTE,),
    )


def estimate_selection_cost(
    spec: SelectionSpec, *, n_features: int, n_folds: int, n_labels: int
) -> CostEstimate:
    """Upper bound of the candidate evaluations of one search, times the folds.

    Forward: with ``r = min(max_features, p)`` rounds over ``p`` candidates the
    search evaluates at most ``sum(p - i for i in range(r))`` candidates.
    Backward: with ``r = p - min_features`` rounds it evaluates at most
    ``r * p`` candidates (each round scores the current set minus one feature),
    plus one reference evaluation per fold for the initial set.
    """
    if n_features < 1 or n_folds < 1 or n_labels < 1:
        raise ValueError("n_features, n_folds and n_labels must be >= 1.")
    settings = spec.config
    if spec.strategy == SELECTION_FORWARD:
        rounds = max(0, min(settings.max_features, n_features))
        candidates = sum(n_features - index for index in range(rounds))
        extra_fits = 0
    else:
        rounds = max(0, n_features - spec.min_features)
        candidates = rounds * n_features
        extra_fits = n_folds
    if settings.max_evaluations is not None:
        candidates = min(candidates, settings.max_evaluations)
    fold_evaluations = candidates * n_folds
    fits = fold_evaluations + extra_fits
    return CostEstimate(
        tasks=1,
        fold_evaluations=fold_evaluations,
        estimator_fits=fits,
        binary_fits=fits * n_labels,
        selection_candidate_evaluations=candidates,
        exact=False,
        notes=(
            SEARCH_UPPER_BOUND_NOTE,
            MAX_EXACT_BINARY_FITS_NOTE,
            "the reference estimator of the search is the one in the selection "
            "configuration, not the model of each evaluation task",
        ),
    )


def dataset_identity(dataset: ProteinDataset) -> str:
    """Deterministic id from metadata only (no feature/label values, no test rows)."""
    return digest(
        {
            "n_samples": dataset.n_samples,
            "feature_names": list(dataset.feature_names),
            "label_names": list(dataset.label_names),
            "accessions": [key.accession for key in dataset.keys],
        }
    )


def folds_identity(plan: SplitPlan) -> str:
    """Deterministic id of the validation folds; the test block is excluded."""
    return digest(
        {
            "seed": plan.seed,
            "test_size": plan.test_size,
            "n_folds": plan.n_folds,
            "folds": [
                {
                    "index": fold.index,
                    "train": [int(value) for value in fold.train],
                    "validation": [int(value) for value in fold.validation],
                }
                for fold in plan.folds
            ],
        }
    )


@dataclass(frozen=True)
class PlannedTask:
    """One schedulable task, with its identity, dependencies and estimated cost."""

    order: int
    task_id: str
    kind: str
    strategy: str
    subset: SubsetDefinition
    cost: CostEstimate
    model_spec_id: str | None = None
    model_label: str | None = None
    selection_id: str | None = None
    requires: tuple[str, ...] = ()
    status: str = STATUS_PLANNED
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status {self.status!r}.")
        if self.kind not in (KIND_SELECTION, KIND_EVALUATION):
            raise ValueError(f"Unknown task kind {self.kind!r}.")
        if self.kind == KIND_EVALUATION and not self.model_spec_id:
            raise ValueError("An evaluation task needs a model_spec_id.")
        if self.status in SKIPPED_STATUSES and not self.skip_reason:
            raise ValueError(f"Status {self.status!r} needs a skip_reason.")

    @property
    def skipped(self) -> bool:
        return self.status in SKIPPED_STATUSES

    def with_status(self, status: str, reason: str | None = None) -> "PlannedTask":
        return replace(self, status=status, skip_reason=reason)

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "task_id": self.task_id,
            "kind": self.kind,
            "strategy": self.strategy,
            "subset": self.subset.as_dict(),
            "model_spec_id": self.model_spec_id,
            "model_label": self.model_label,
            "selection_id": self.selection_id,
            "requires": list(self.requires),
            "cost": self.cost.as_dict(),
            "status": self.status,
            "skip_reason": self.skip_reason,
        }


def _task_id(**identity: object) -> str:
    return digest({"protocol": PROTOCOL_VERSION, **identity}, length=20)


def _validate_plan_shape(plan: SplitPlan, n_samples: int) -> None:
    """Check fold indices without touching ``plan.test``."""
    if plan.n_folds < 1:
        raise ValueError(f"plan has no folds (n_folds={plan.n_folds}).")
    validated: list[int] = []
    for fold in plan.folds:
        for name, block in (("train", fold.train), ("validation", fold.validation)):
            values = [int(value) for value in block]
            if not values:
                raise ValueError(f"{fold.name} {name} is empty.")
            if min(values) < 0 or max(values) >= n_samples:
                raise ValueError(
                    f"{fold.name} {name} has an index outside [0, {n_samples})."
                )
        train = {int(value) for value in fold.train}
        validation = {int(value) for value in fold.validation}
        if train & validation:
            raise ValueError(f"{fold.name} shares indices between train and validation.")
        validated.extend(sorted(validation))
    if len(set(validated)) != len(validated):
        raise ValueError("A sample is validated in more than one fold.")


@dataclass(frozen=True)
class ExecutionPlan:
    """The enumerated tasks of one experiment, inspectable without training."""

    config_id: str
    dataset_id: str
    folds_id: str
    protocol_version: str
    seed: int
    n_samples: int
    n_features: int
    n_labels: int
    n_folds: int
    tasks: tuple[PlannedTask, ...]
    notes: tuple[str, ...] = ()

    @property
    def plan_id(self) -> str:
        return digest(
            {
                "protocol": self.protocol_version,
                "config_id": self.config_id,
                "dataset_id": self.dataset_id,
                "folds_id": self.folds_id,
                "tasks": [task.task_id for task in self.tasks],
            }
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def get(self, task_id: str) -> PlannedTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown task_id {task_id!r}.")

    def by_status(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES if status != STATUS_RUNNING}
        for task in self.tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def tasks_with_status(self, *statuses: str) -> tuple[PlannedTask, ...]:
        return tuple(task for task in self.tasks if task.status in statuses)

    def total_cost(self, *, include_skipped: bool = False) -> CostEstimate:
        selected = [
            task.cost for task in self.tasks if include_skipped or not task.skipped
        ]
        return total_cost(selected)

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "protocol_version": self.protocol_version,
            "config_id": self.config_id,
            "dataset_id": self.dataset_id,
            "folds_id": self.folds_id,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "n_folds": self.n_folds,
            "n_tasks": len(self.tasks),
            "status_counts": self.by_status(),
            "estimated_cost": self.total_cost().as_dict(),
            "tasks": [task.as_dict() for task in self.tasks],
            "notes": list(self.notes),
        }


def build_execution_plan(
    config: ExperimentConfig,
    dataset: ProteinDataset,
    split_plan: SplitPlan,
    *,
    dependency_available: Callable[[str], bool] | None = None,
) -> ExecutionPlan:
    """Enumerate the tasks of ``config`` without training anything.

    Args:
        config: A validated :class:`ExperimentConfig`.
        dataset: Used for metadata only (shapes, names, accessions).
        split_plan: The approved folds; only ``fold.train``/``fold.validation``
            indices are read, never ``plan.test``.
        dependency_available: Probe for optional dependencies, called with the
            module name. Defaults to ``importlib.util.find_spec``, which does
            not import the module.

    Raises:
        DatasetError: If the dataset is invalid.
        ValueError: On invalid fold indices.

    Returns:
        An :class:`ExecutionPlan`: one task per selection configuration plus one
        per (strategy, model configuration) pair.
    """
    probe = dependency_available or default_dependency_probe
    validate_dataset(dataset)
    _validate_plan_shape(split_plan, dataset.n_samples)

    dataset_id = dataset_identity(dataset)
    folds_id = folds_identity(split_plan)
    evaluation_id = digest(selection_parameters(config.evaluation, min_features=1))
    n_folds = split_plan.n_folds
    n_features = dataset.n_features
    n_labels = dataset.n_labels

    tasks: list[PlannedTask] = []
    selection_ids: dict[str, str] = {}

    for spec in config.selection:
        task_id = _task_id(
            kind=KIND_SELECTION,
            strategy=spec.strategy,
            selection_id=spec.spec_id,
            dataset_id=dataset_id,
            folds_id=folds_id,
            evaluation_id=evaluation_id,
        )
        selection_ids[spec.strategy] = task_id
        tasks.append(
            PlannedTask(
                order=len(tasks),
                task_id=task_id,
                kind=KIND_SELECTION,
                strategy=spec.strategy,
                subset=SubsetDefinition(SUBSET_SELECTION, spec.spec_id),
                cost=estimate_selection_cost(
                    spec, n_features=n_features, n_folds=n_folds, n_labels=n_labels
                ),
                selection_id=spec.spec_id,
            )
        )

    for strategy in config.strategies:
        if strategy == STRATEGY_ALL:
            subset = SubsetDefinition(SUBSET_ALL)
            requires: tuple[str, ...] = ()
        elif strategy == STRATEGY_VARIANCE:
            subset = SubsetDefinition(SUBSET_VARIANCE)
            requires = ()
        elif strategy in (STRATEGY_FORWARD, STRATEGY_BACKWARD):
            spec = config.selection_spec(strategy)
            if spec is None:  # pragma: no cover - guarded by config validation
                raise ValueError(f"Strategy {strategy!r} has no SelectionSpec.")
            subset = SubsetDefinition(SUBSET_SELECTION, spec.spec_id)
            requires = (selection_ids[strategy],)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown strategy {strategy!r}.")

        for model in config.models:
            task_id = _task_id(
                kind=KIND_EVALUATION,
                strategy=strategy,
                subset=subset.as_dict(),
                model_spec_id=model.spec_id,
                dataset_id=dataset_id,
                folds_id=folds_id,
                evaluation_id=evaluation_id,
            )
            status, reason = _availability(model, probe)
            tasks.append(
                PlannedTask(
                    order=len(tasks),
                    task_id=task_id,
                    kind=KIND_EVALUATION,
                    strategy=strategy,
                    subset=subset,
                    cost=estimate_evaluation_cost(
                        n_folds=n_folds, n_labels=n_labels
                    ),
                    model_spec_id=model.spec_id,
                    model_label=model.label,
                    requires=requires,
                    status=status,
                    skip_reason=reason,
                )
            )

    return ExecutionPlan(
        config_id=config.config_id,
        dataset_id=dataset_id,
        folds_id=folds_id,
        protocol_version=config.protocol_version,
        seed=config.seed,
        n_samples=dataset.n_samples,
        n_features=n_features,
        n_labels=n_labels,
        n_folds=n_folds,
        tasks=tuple(tasks),
        notes=(
            "the external test block is not part of this plan and is never read",
            "estimates are upper bounds, not measurements",
            "task ids depend on the protocol, the configuration, the dataset "
            "metadata, the folds and the evaluation policy; they never depend "
            "on test-block results",
        ),
    )


def _availability(
    model: ModelSpec, probe: Callable[[str], bool]
) -> tuple[str, str | None]:
    module = model.requires_dependency
    if module and not probe(module):
        return (
            STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
            f"optional dependency {module!r} is not installed; install it "
            "explicitly to run this configuration",
        )
    return STATUS_PLANNED, None
