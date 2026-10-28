"""Manifest and final report builders (pure serialisation, no execution).

Everything here is JSON-ready. Two rules are enforced by construction:

* the manifest carries the *identity* (protocol, configuration, dataset, folds,
  ordered task ids) plus metadata that must never influence it (``created_at``,
  environment versions);
* the report never invents values. A task without a validated result has
  ``result: null`` and, when nothing ran at all, the whole report says
  ``results_available: false`` instead of a table of zeros.

No external-test metric can appear here: nothing in this phase can produce one.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from protein_database.ml.dataset import ProteinDataset
from protein_database.ml.experiment_budget import RunBudget
from protein_database.ml.experiment_config import (
    PROTOCOL_VERSION,
    ExperimentConfig,
    describe_catalog,
)
from protein_database.ml.experiment_plan import (
    STATUS_COMPLETED,
    ExecutionPlan,
    dataset_identity,
    folds_identity,
)
from protein_database.ml.splits import SplitPlan

MANIFEST_VERSION = 1
REPORT_VERSION = 1
ENVIRONMENT_PACKAGES = ("scikit-learn", "numpy", "scipy", "pandas")
OPTIONAL_PACKAGES = ("xgboost",)

POLICY_DECISIONS = {
    "folds": "one SplitPlan for every model and strategy; no per-model folds",
    "external_test": "never read, never indexed and absent from every identifier",
    "aggregation": "macro-F1 per fold, then mean and population std over folds",
    "threshold": "fixed 0.5 decision threshold; no threshold tuning",
    "label_masks": "fit needs >=2 positives and >=2 negatives, eval needs >=1 "
    "positive, intersected per fold; excluded labels are reported",
    "variance_prefilter": "eligible features are those kept by the variance "
    "filter of every fold; the filter is fitted on the fold's training block",
    "selection": "one selection run per selection configuration, reused by the "
    "evaluations that depend on it; no nested search per model",
    "duplicates": "identical model configurations are deduplicated (first label "
    "kept and recorded); distinct configurations of one family are kept apart",
    "convergence": "a ConvergenceWarning marks the task completed with "
    "converged=false; it is not an error",
    "budget_time": "cooperative: checked before each task, never interrupts a "
    "running estimator",
    "costs": "estimates are upper bounds; observed values come from the "
    "evaluation and selection objects",
}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict:
    """Versions available without network access and without importing extras."""
    packages = {name: _package_version(name) for name in ENVIRONMENT_PACKAGES}
    optional = {name: _package_version(name) for name in OPTIONAL_PACKAGES}
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "optional_packages": optional,
        "optional_available": {
            name: version is not None for name, version in optional.items()
        },
        "note": "recorded as metadata; it is not part of any identifier",
    }


def timestamp() -> str:
    """Creation time as ISO-8601 UTC (metadata only, never part of an id)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plan_summary(plan: SplitPlan) -> dict:
    return {
        "folds_id": folds_identity(plan),
        "n_folds": plan.n_folds,
        "seed": plan.seed,
        "test_size": plan.test_size,
        "n_groups_test": plan.n_groups_test,
        "n_groups_dev": plan.n_groups_dev,
        "folds": [
            {
                "name": fold.name,
                "n_train": int(fold.train.size),
                "n_validation": int(fold.validation.size),
            }
            for fold in plan.folds
        ],
        "test_block": "excluded from the plan and from every identifier",
    }


def build_manifest(
    config: ExperimentConfig,
    execution_plan: ExecutionPlan,
    *,
    dataset: ProteinDataset,
    split_plan: SplitPlan,
    budget: RunBudget,
    created_at: str | None = None,
    environment: dict | None = None,
) -> dict:
    """The reproduction record of one planned experiment."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "config_id": config.config_id,
        "plan_id": execution_plan.plan_id,
        "dataset": {
            "dataset_id": dataset_identity(dataset),
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "n_labels": dataset.n_labels,
            "feature_names": list(dataset.feature_names),
            "label_names": list(dataset.label_names),
        },
        "split_plan": _plan_summary(split_plan),
        "seed": config.seed,
        "environment": environment if environment is not None else environment_info(),
        "budget": budget.as_dict(),
        "catalog": describe_catalog(config),
        "configuration": config.effective(),
        "policy": dict(POLICY_DECISIONS),
        "plan": execution_plan.as_dict(),
        "created_at": created_at or timestamp(),
        "identity_excludes": ["created_at", "environment"],
    }


def write_json(path: str | Path, payload: object) -> Path:
    """Write deterministic JSON (sorted keys) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def result_rows(tasks: list[dict]) -> list[dict]:
    """One row per evaluation task that produced a result (never a filled-in one)."""
    rows: list[dict] = []
    for task in tasks:
        result = task.get("result")
        if task["status"] != STATUS_COMPLETED or not result:
            continue
        if task["kind"] != "evaluation":
            continue
        rows.append(
            {
                "model_label": task.get("model_label"),
                "model_spec_id": task.get("model_spec_id"),
                "strategy": task["strategy"],
                "subset": task["subset"]["kind"],
                "n_features": result.get("n_features"),
                "mean_macro_f1": result.get("mean_macro_f1"),
                "std_macro_f1": result.get("std_macro_f1"),
                "n_folds": len(result.get("folds") or []),
                "converged": result.get("converged"),
                "excluded_labels": result.get("excluded_labels", []),
                "estimated": task.get("estimated"),
                "observed": task.get("observed"),
            }
        )
    return rows


def build_report(
    *,
    config: ExperimentConfig,
    execution_plan: ExecutionPlan,
    tasks: list[dict],
    ledger_snapshot: dict,
    estimated_total: dict,
    created_at: str | None = None,
    notes: tuple[str, ...] = (),
) -> dict:
    """Final, JSON-ready report of what happened to each planned task."""
    counts = {
        "planned": 0,
        "completed": 0,
        "failed": 0,
        "skipped_budget": 0,
        "skipped_unavailable_dependency": 0,
        "skipped_dependency_failure": 0,
    }
    for task in tasks:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    rows = result_rows(tasks)
    return {
        "report_version": REPORT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "config_id": config.config_id,
        "plan_id": execution_plan.plan_id,
        "dataset_id": execution_plan.dataset_id,
        "folds_id": execution_plan.folds_id,
        "results_available": bool(rows),
        "results_statement": (
            "validated results available for the completed tasks"
            if rows
            else "no results available: nothing was executed or nothing completed"
        ),
        "status_counts": counts,
        "n_planned": len(tasks),
        "rows": rows,
        "tasks": tasks,
        "estimated_cost_total": estimated_total,
        "observed_cost": ledger_snapshot,
        "external_test": "not evaluated in this phase; no external metric exists",
        "notes": list(POLICY_DECISIONS.values()) + list(notes),
        "created_at": created_at or timestamp(),
    }
