"""Plan-then-execute orchestration: states, incremental storage, final report.

The runner **only** executes a plan it built itself; it never decides what to run
while running. Each task is charged against the budget *before* it starts, its
observed cost is charged after, and every finished task is appended to a JSONL
store immediately, so an interrupted run leaves a readable partial record.

Nothing here is allowed to touch the external test block: the dataset handed to
the evaluators is wrapped by :func:`protein_database.ml.experiment_guard.guard_dataset`,
which raises :class:`~protein_database.ml.experiment_guard.TestBlockAccessError`
on any row selection that intersects ``plan.test``.

States: ``planned``, ``completed``, ``failed``, ``skipped_budget``,
``skipped_unavailable_dependency``, ``skipped_dependency_failure``. ``running``
exists as a constant but is never persisted, and ``cancelled`` does not exist
because no safe cancellation is implemented.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from protein_database.ml.dataset import ProteinDataset
from protein_database.ml.experiment_budget import BudgetLedger, RunBudget
from protein_database.ml.experiment_config import (
    ON_ERROR_ABORT,
    SELECTION_FORWARD,
    ExperimentConfig,
    spec_parameters,
)
from protein_database.ml.experiment_guard import guard_dataset
from protein_database.ml.experiment_plan import (
    KIND_SELECTION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PLANNED,
    STATUS_SKIPPED_BUDGET,
    STATUS_SKIPPED_DEPENDENCY_FAILURE,
    STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
    STATUSES,
    SUBSET_ALL,
    SUBSET_VARIANCE,
    ExecutionPlan,
    PlannedTask,
    build_execution_plan,
    default_dependency_probe,
)
from protein_database.ml.experiment_report import (
    build_manifest,
    build_report,
    timestamp,
    write_json,
)
from protein_database.ml.feature_selection import (
    eligible_features,
    evaluate_feature_subset,
    run_backward_elimination,
    run_forward_selection,
)
from protein_database.ml.models import MissingDependencyError
from protein_database.ml.splits import SplitPlan

STORE_VERSION = 1
STORE_HEADER_RECORD = "header"
STORE_TASK_RECORD = "task"


class ResumeMismatchError(RuntimeError):
    """A stored run belongs to a different protocol, configuration or plan."""


@dataclass(frozen=True)
class TaskResult:
    """State and payload of one task after the runner touched it."""

    task_id: str
    status: str
    observed: Mapping[str, int] = field(default_factory=dict)
    result: Mapping | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
    skip_reason: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status {self.status!r}.")
        if self.status == STATUS_COMPLETED and self.result is None:
            raise ValueError("A completed task must carry a result.")
        if self.status == STATUS_FAILED and not self.error:
            raise ValueError("A failed task must carry an error message.")
        if self.status in (
            STATUS_SKIPPED_BUDGET,
            STATUS_SKIPPED_DEPENDENCY_FAILURE,
            STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
        ) and not self.skip_reason:
            raise ValueError(f"Status {self.status!r} needs a skip_reason.")

    @property
    def executed(self) -> bool:
        return self.status in (STATUS_COMPLETED, STATUS_FAILED)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": None if self.result is None else dict(self.result),
            "observed": dict(self.observed),
            "error": self.error,
            "warnings": list(self.warnings),
            "skip_reason": self.skip_reason,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskResult":
        return cls(
            task_id=str(payload["task_id"]),
            status=str(payload["status"]),
            observed=dict(payload.get("observed") or {}),
            result=payload.get("result"),
            error=payload.get("error"),
            warnings=tuple(payload.get("warnings") or ()),
            skip_reason=payload.get("skip_reason"),
            note=payload.get("note"),
        )


class TaskStore:
    """Append-only JSONL store of finished tasks, with a validating header."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def identity_fields(self) -> tuple[str, ...]:
        return ("protocol_version", "config_id", "dataset_id", "folds_id", "plan_id")

    def read(self) -> tuple[dict | None, dict[str, dict]]:
        """Return ``(header, {task_id: record})``; missing file gives ``(None, {})``."""
        if not self.path.exists():
            return None, {}
        header: dict | None = None
        records: dict[str, dict] = {}
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{self.path}: line {number} is not valid JSON ({error.msg})."
                ) from error
            kind = payload.get("record")
            if kind == STORE_HEADER_RECORD:
                header = payload
            elif kind == STORE_TASK_RECORD:
                records[str(payload["task_id"])] = payload
            else:
                raise ValueError(
                    f"{self.path}: line {number} has unknown record type {kind!r}."
                )
        return header, records

    def write_header(self, identity: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "record": STORE_HEADER_RECORD,
            "store_version": STORE_VERSION,
            **dict(identity),
            "created_at": timestamp(),
        }
        self.path.write_text(
            json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def append(self, payload: Mapping[str, object]) -> None:
        """Append one task record and flush, so a crash keeps prior results."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"record": STORE_TASK_RECORD, **dict(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()

    def load(self, identity: Mapping[str, object]) -> dict[str, dict]:
        """Load reusable records, refusing a store from another run."""
        header, records = self.read()
        if header is None:
            return {}
        mismatch = {
            name: (header.get(name), identity.get(name))
            for name in self.identity_fields()
            if header.get(name) != identity.get(name)
        }
        if mismatch:
            details = ", ".join(
                f"{name}: stored={stored!r} expected={expected!r}"
                for name, (stored, expected) in mismatch.items()
            )
            raise ResumeMismatchError(
                "Stored results belong to a different run and cannot be reused "
                f"({details})."
            )
        return records


class ExperimentRunner:
    """Builds a plan, then executes it on request, enforcing the budget."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: ProteinDataset,
        split_plan: SplitPlan,
        *,
        budget: RunBudget | None = None,
        dependency_available: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.split_plan = split_plan
        self.budget = budget or RunBudget()
        self.dependency_available = dependency_available or default_dependency_probe
        self.clock = clock
        self._plan: ExecutionPlan | None = None

    # -- planning -----------------------------------------------------------

    def plan(self) -> ExecutionPlan:
        """The cached execution plan; planning never trains anything."""
        if self._plan is None:
            self._plan = build_execution_plan(
                self.config,
                self.dataset,
                self.split_plan,
                dependency_available=self.dependency_available,
            )
        return self._plan

    def store_identity(self) -> dict:
        plan = self.plan()
        return {
            "protocol_version": plan.protocol_version,
            "config_id": plan.config_id,
            "dataset_id": plan.dataset_id,
            "folds_id": plan.folds_id,
            "plan_id": plan.plan_id,
        }

    def manifest(self, *, created_at: str | None = None) -> dict:
        return build_manifest(
            self.config,
            self.plan(),
            dataset=self.dataset,
            split_plan=self.split_plan,
            budget=self.budget,
            created_at=created_at,
        )

    # -- execution ----------------------------------------------------------

    def run(
        self,
        *,
        store_path: str | Path | None = None,
        report_path: str | Path | None = None,
        resume: bool = True,
        created_at: str | None = None,
        write_artifacts: bool = True,
    ) -> dict:
        """Execute the plan in order and return the final report.

        Args:
            store_path: JSONL file for incremental task records. Existing
                records are reused only if the header matches this run exactly.
            report_path: Where to write the report. Defaults to
                ``<store>-report.json`` when a store is given.
            resume: Reuse compatible records from ``store_path``.
            created_at: Fixed timestamp for reproducible artifacts (metadata).
            write_artifacts: When ``False`` nothing is written, not even for a
                given ``store_path``; used by tests that only inspect results.

        Raises:
            ResumeMismatchError: If the store belongs to another run.
        """
        plan = self.plan()
        store = TaskStore(store_path) if store_path is not None else None
        reused: dict[str, dict] = {}
        if store is not None:
            reused = store.load(self.store_identity()) if resume else {}
            if not store.path.exists() or not resume:
                store.write_header(self.store_identity())

        guarded = guard_dataset(self.dataset, self.split_plan)
        ledger = BudgetLedger(budget=self.budget, clock=self.clock)
        results: dict[str, TaskResult] = {}
        selected_features: dict[str, tuple[str, ...]] = {}
        abort_reason: str | None = None

        for task in plan.tasks:
            if abort_reason is not None:
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status=STATUS_PLANNED,
                    note=f"not started: {abort_reason}",
                )
                continue
            if task.skipped:
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status=task.status,
                    skip_reason=task.skip_reason,
                )
                continue
            if task.task_id in reused:
                stored = reused[task.task_id]
                if stored.get("status") == STATUS_COMPLETED:
                    entry = TaskResult.from_dict(stored)
                    # a reused task already paid for itself: charge its recorded cost
                    ledger.charge_observed(entry.observed)
                    results[task.task_id] = entry
                    self._remember_selection(task, entry, selected_features)
                    continue

            blocked = self._blocked_dependency(task, results)
            if blocked is not None:
                entry = TaskResult(
                    task_id=task.task_id,
                    status=STATUS_SKIPPED_DEPENDENCY_FAILURE,
                    skip_reason=blocked,
                )
            else:
                affordable, reason = ledger.can_afford(task.cost)
                if not affordable:
                    entry = TaskResult(
                        task_id=task.task_id,
                        status=STATUS_SKIPPED_BUDGET,
                        skip_reason=reason,
                    )
                else:
                    entry = self._execute(task, guarded, selected_features)
                    if entry.status == STATUS_FAILED and (
                        self.config.failure_policy.on_task_error == ON_ERROR_ABORT
                    ):
                        abort_reason = (
                            f"run aborted after task {task.task_id} failed "
                            f"({entry.error})"
                        )

            if entry.observed:
                # only tasks that really ran are charged; a skipped task costs nothing
                ledger.charge_observed(entry.observed)
            results[task.task_id] = entry
            self._remember_selection(task, entry, selected_features)
            if store is not None:
                store.append(entry.as_dict())

        task_rows = self._task_rows(plan, results)
        estimated = plan.total_cost()
        report = build_report(
            config=self.config,
            execution_plan=plan,
            tasks=task_rows,
            ledger_snapshot=ledger.snapshot(),
            estimated_total=estimated.as_dict(),
            created_at=created_at,
        )

        if store is not None and write_artifacts:
            write_json(self._manifest_path(store.path), self.manifest(created_at=created_at))
        if write_artifacts and (report_path is not None or store is not None):
            target = (
                Path(report_path)
                if report_path is not None
                else self._report_path_for(store.path)  # type: ignore[union-attr]
            )
            write_json(target, report)
        return report

    # -- task execution -----------------------------------------------------

    def _execute(
        self,
        task: PlannedTask,
        dataset: ProteinDataset,
        selected_features: Mapping[str, tuple[str, ...]],
    ) -> TaskResult:
        try:
            if task.kind == KIND_SELECTION:
                payload, observed = self._run_selection(task, dataset)
            else:
                payload, observed = self._run_evaluation(
                    task, dataset, selected_features
                )
        except MissingDependencyError as error:
            return TaskResult(
                task_id=task.task_id,
                status=STATUS_SKIPPED_UNAVAILABLE_DEPENDENCY,
                observed={"tasks": 1},
                skip_reason=str(error),
            )
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            return TaskResult(
                task_id=task.task_id,
                status=STATUS_FAILED,
                observed={"tasks": 1},
                error=f"{type(error).__name__}: {error}",
            )
        warnings: tuple[str, ...] = ()
        if payload.get("converged") is False:
            warnings = (
                "ConvergenceWarning: at least one fold did not converge; the "
                "task completed and the score is reported as-is",
            )
        return TaskResult(
            task_id=task.task_id,
            status=STATUS_COMPLETED,
            observed=observed,
            result=payload,
            warnings=warnings,
        )

    def _run_selection(
        self, task: PlannedTask, dataset: ProteinDataset
    ) -> tuple[dict, dict[str, int]]:
        spec = self.config.selection_spec(task.strategy)
        if spec is None:  # pragma: no cover - plan and config are consistent
            raise KeyError(f"No selection configuration for {task.strategy!r}.")
        if task.strategy == SELECTION_FORWARD:
            outcome = run_forward_selection(
                dataset, self.split_plan, config=spec.config
            )
            payload = outcome.report()
            selected = tuple(payload["selected"])
        else:
            outcome = run_backward_elimination(
                dataset,
                self.split_plan,
                config=spec.config,
                min_features=spec.min_features,
            )
            payload = outcome.report()
            selected = tuple(payload["remaining"])
        payload = {
            "strategy": task.strategy,
            "selection_id": spec.spec_id,
            "n_selected": len(selected),
            "selected": list(selected),
            "converged": _all_converged(payload.get("history") or []),
            **payload,
        }
        # wall-clock per round is measurement noise, not a result: drop it so two
        # identical runs store identical payloads (the run elapsed time is kept
        # once, in the budget snapshot)
        payload["history"] = _without_timings(payload.get("history") or [])
        observed = {
            "tasks": 1,
            "fold_evaluations": int(payload["n_evaluations"]),
            "estimator_fits": int(payload["n_evaluations"]),
            "binary_fits": int(payload["n_model_fits"]),
            "selection_candidate_evaluations": int(payload["n_evaluations"]),
        }
        return payload, observed

    def _run_evaluation(
        self,
        task: PlannedTask,
        dataset: ProteinDataset,
        selected_features: Mapping[str, tuple[str, ...]],
    ) -> tuple[dict, dict[str, int]]:
        spec = self._model_spec(task.model_spec_id)
        features = self._features_for(task, dataset, selected_features)
        evaluation = evaluate_feature_subset(
            dataset,
            self.split_plan,
            features,
            config=self.config.evaluation,
            model=spec.config,
        )
        folds = [
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
            for score in evaluation.fold_scores
        ]
        excluded = sorted(
            {
                label
                for score in evaluation.fold_scores
                for label in score.excluded_labels
            }
        )
        payload = {
            "model_spec_id": spec.spec_id,
            "model_label": spec.label,
            "model_id": spec.model_id,
            "strategy": task.strategy,
            "subset": task.subset.as_dict(),
            "features": list(evaluation.features),
            "n_features": len(evaluation.features),
            "mean_macro_f1": evaluation.mean_macro_f1,
            "std_macro_f1": evaluation.std_macro_f1,
            "converged": evaluation.converged,
            "excluded_folds": list(evaluation.excluded_folds),
            "excluded_labels": excluded,
            "n_estimator_fits": len(evaluation.fold_scores),
            "n_binary_fits": evaluation.n_model_fits,
            "folds": folds,
            "estimator": spec_parameters(spec.config),
        }
        observed = {
            "tasks": 1,
            "fold_evaluations": len(evaluation.fold_scores),
            "estimator_fits": len(evaluation.fold_scores),
            "binary_fits": int(evaluation.n_model_fits),
            "selection_candidate_evaluations": 0,
        }
        return payload, observed

    # -- helpers ------------------------------------------------------------

    def _model_spec(self, spec_id: str | None):
        for spec in self.config.models:
            if spec.spec_id == spec_id:
                return spec
        raise KeyError(f"Unknown model spec_id {spec_id!r}.")  # pragma: no cover

    def _features_for(
        self,
        task: PlannedTask,
        dataset: ProteinDataset,
        selected_features: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        if task.subset.kind == SUBSET_ALL:
            return tuple(dataset.feature_names)
        if task.subset.kind == SUBSET_VARIANCE:
            return tuple(
                eligible_features(
                    dataset, self.split_plan, config=self.config.evaluation
                )
            )
        features = selected_features.get(task.subset.selection_id or "")
        if features is None:  # pragma: no cover - guarded by the dependency check
            raise RuntimeError(
                "Selection subset requested before its selection task completed."
            )
        return features

    def _blocked_dependency(
        self, task: PlannedTask, results: Mapping[str, TaskResult]
    ) -> str | None:
        for dependency in task.requires:
            entry = results.get(dependency)
            if entry is None or entry.status != STATUS_COMPLETED:
                status = None if entry is None else entry.status
                return (
                    f"required task {dependency} did not complete "
                    f"(status={status!r}); no substitute subset is created"
                )
        return None

    def _remember_selection(
        self,
        task: PlannedTask,
        entry: TaskResult,
        selected_features: dict[str, tuple[str, ...]],
    ) -> None:
        if (
            task.kind == KIND_SELECTION
            and entry.status == STATUS_COMPLETED
            and entry.result
            and task.selection_id
        ):
            selected_features[task.selection_id] = tuple(entry.result["selected"])

    def _task_rows(
        self, plan: ExecutionPlan, results: Mapping[str, TaskResult]
    ) -> list[dict]:
        rows: list[dict] = []
        for task in plan.tasks:
            entry = results[task.task_id]
            planned = task.as_dict()
            estimated = planned.pop("cost")
            rows.append({**planned, "estimated": estimated, **entry.as_dict()})
        return rows

    @staticmethod
    def _manifest_path(store_path: Path) -> Path:
        return Path(f"{store_path}.manifest.json")

    @staticmethod
    def _report_path_for(store_path: Path) -> Path:
        return Path(f"{store_path}.report.json")


def _all_converged(history: Sequence[Mapping[str, object]]) -> bool | None:
    flags = [row.get("converged") for row in history]
    if not flags or any(flag is None for flag in flags):
        return None
    return all(bool(flag) for flag in flags)


def _without_timings(
    history: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Copy of the selection history without per-round wall-clock times."""
    return [
        {key: value for key, value in row.items() if key != "elapsed_seconds"}
        for row in history
    ]

