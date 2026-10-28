"""One explicit experimental flow: feature strategies x models, development only.

What this module guarantees
---------------------------
* :func:`development_block` returns a dataset that physically contains only the
  development rows, with the folds re-indexed onto it. Selection, tuning and
  model comparison run on that object, so they cannot read an external test row:
  no test row is in the arrays they receive. This is structural, not an access
  guard that has to intercept every read.
* :func:`evaluate_external` is the single function here that indexes
  ``plan.test``. :func:`run_experiment` never calls it: the external block is
  scored once, by an explicit call, after the configuration was chosen.
* Dataset validation (shapes, finiteness, duplicates) runs on the full matrix
  before the split, so it does inspect test values; it never uses them to
  choose, rank or tune anything.

Result reuse
------------
``results.json`` stores the settings and a fingerprint of the effective ``X`` and
``y`` values. A later run reuses a stored row only when the fingerprint matches,
so changing the features, the labels or the sample changes the fingerprint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from protein_database.ml.dataset import ProteinDataset, validate_dataset
from protein_database.ml.evaluation import (
    evaluable_label_mask,
    fit_label_mask,
    multilabel_metrics,
)
from protein_database.ml.feature_selection import (
    BackwardEliminationResult,
    ForwardSelectionConfig,
    ForwardSelectionResult,
    eligible_features,
    run_backward_elimination,
    run_forward_selection,
)
from protein_database.ml.model_comparison import compare_models
from protein_database.ml.models import (
    LogisticConfig,
    RandomForestConfig,
    XGBoostConfig,
    build_estimator,
    describe_estimator,
)
from protein_database.ml.splits import Fold, SplitPlan
from protein_database.ml.strategies import (
    STRATEGIES,
    STRATEGY_ALL,
    STRATEGY_BACKWARD,
    STRATEGY_FORWARD,
    STRATEGY_VARIANCE,
)

STORAGE_VERSION = 1

#: One configuration per family: the comparison the project needs.
DEFAULT_MODELS = (LogisticConfig(), RandomForestConfig(), XGBoostConfig())


@dataclass(frozen=True)
class ExperimentSettings:
    """What to compare, and with which parameters."""

    models: tuple = DEFAULT_MODELS
    strategies: tuple[str, ...] = STRATEGIES
    forward: ForwardSelectionConfig = field(
        default_factory=lambda: ForwardSelectionConfig(max_features=10)
    )
    backward: ForwardSelectionConfig = field(
        default_factory=lambda: ForwardSelectionConfig(max_features=10)
    )
    backward_min_features: int = 2
    evaluation: ForwardSelectionConfig = field(default_factory=ForwardSelectionConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        unknown = [name for name in self.strategies if name not in STRATEGIES]
        if unknown:
            raise ValueError(
                f"Unknown strategy/strategies {unknown}; expected a subset of {STRATEGIES}."
            )
        if not self.strategies:
            raise ValueError("strategies is empty: nothing to compare.")
        if not self.models:
            raise ValueError("models is empty: nothing to compare.")
        families = [describe_estimator(model)["model_id"] for model in self.models]
        if len(set(families)) != len(families):
            raise ValueError(
                f"Two configurations of the same family: {families}. The report "
                "identifies a model by its family, so one configuration per family is used."
            )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer; got {self.seed!r}.")

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "strategies": list(self.strategies),
            "models": [describe_estimator(model) for model in self.models],
            "forward": selection_dict(self.forward),
            "backward": selection_dict(
                self.backward, min_features=self.backward_min_features
            ),
            "evaluation": selection_dict(self.evaluation),
        }


def selection_dict(
    config: ForwardSelectionConfig, *, min_features: int | None = None
) -> dict:
    """Serialisable selection parameters, defaults included."""
    payload = {
        "max_features": config.max_features,
        "min_improvement": config.min_improvement,
        "max_evaluations": config.max_evaluations,
        "variance_threshold": config.variance_threshold,
        "min_positives_fit": config.min_positives_fit,
        "min_negatives_fit": config.min_negatives_fit,
        "min_positives_eval": config.min_positives_eval,
        "estimator": describe_estimator(config.estimator),
    }
    if min_features is not None:
        payload["min_features"] = min_features
    return payload


def development_block(
    dataset: ProteinDataset, plan: SplitPlan
) -> tuple[ProteinDataset, SplitPlan]:
    """Copy the development rows and re-index the folds onto the copy.

    Returns a dataset that contains no external test row and a plan whose folds
    index that dataset and whose ``test`` block is empty. Every selection and
    evaluation step downstream uses these two objects.

    Raises:
        ValueError: If the plan has no fold, or a fold index is out of range.
    """
    if not plan.folds:
        raise ValueError("The split plan has no fold.")
    keep = np.sort(np.asarray(plan.dev, dtype=int))
    if keep.size == 0:
        raise ValueError("The split plan has no development row.")
    if keep[0] < 0 or keep[-1] >= dataset.n_samples:
        raise ValueError(
            f"Development indices must be inside [0, {dataset.n_samples}); got "
            f"[{int(keep[0])}, {int(keep[-1])}]."
        )
    position = {int(index): slot for slot, index in enumerate(keep)}
    folds = tuple(
        Fold(
            fold.index,
            np.array([position[int(i)] for i in fold.train], dtype=int),
            np.array([position[int(i)] for i in fold.validation], dtype=int),
        )
        for fold in plan.folds
    )
    sliced = replace(
        dataset,
        keys=tuple(dataset.keys[int(index)] for index in keep),
        X=dataset.X[keep],
        y=dataset.y[keep],
    )
    dev_plan = SplitPlan(
        test=np.array([], dtype=int),
        folds=folds,
        seed=plan.seed,
        test_size=plan.test_size,
        n_groups_test=0,
        n_groups_dev=plan.n_groups_dev,
    )
    return sliced, dev_plan


def data_fingerprint(
    dataset: ProteinDataset, settings: ExperimentSettings
) -> str:
    """SHA-256 of the effective ``X`` and ``y`` values plus the settings.

    The matrices are hashed, so two samples with the same accessions but
    different feature or label values get different fingerprints. The cost is
    one pass over the matrices at the start of a run, never during planning.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset.X, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(dataset.y, dtype=np.bool_).tobytes())
    names = json.dumps(
        [list(dataset.feature_names), list(dataset.label_names)],
        ensure_ascii=False,
    )
    digest.update(names.encode("utf-8"))
    digest.update(
        json.dumps(settings.as_dict(), sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


def run_selection(
    dataset: ProteinDataset,
    plan: SplitPlan,
    settings: ExperimentSettings,
    strategy: str,
) -> dict:
    """Run one search strategy on the development folds and return its audit.

    Only ``forward`` and ``backward`` reach this function; the two baselines are
    plain subsets. Nothing here reads the external test block.
    """
    if strategy == STRATEGY_FORWARD:
        result: ForwardSelectionResult = run_forward_selection(
            dataset, plan, config=settings.forward
        )
        payload = result.report()
        selected = list(result.selection.selected)
    elif strategy == STRATEGY_BACKWARD:
        backward: BackwardEliminationResult = run_backward_elimination(
            dataset,
            plan,
            config=settings.backward,
            min_features=settings.backward_min_features,
        )
        payload = backward.report()
        selected = list(backward.remaining)
    else:
        raise ValueError(f"{strategy!r} is not a search strategy.")
    return {
        "strategy": strategy,
        "selected": selected,
        "n_selected": len(selected),
        "stopped_reason": payload["stopped_reason"],
        "best_score": payload["best_score"],
        "n_evaluations": payload["n_evaluations"],
        "n_rounds": len(payload["history"]),
        "excluded_folds": payload["excluded_folds"],
        "history": payload["history"],
        "fold_label_coverage": payload["fold_label_coverage"],
    }


def evaluation_rows(comparison) -> list[dict]:
    """Flat rows of a model comparison, one per (subset, model)."""
    rows: list[dict] = []
    for item in comparison.evaluations:
        rows.append(
            {
                "strategy": item.subset,
                "model_id": item.model_id,
                "model": dict(item.model),
                "n_features": item.n_features,
                "features": list(item.features),
                "mean_macro_f1": item.mean_macro_f1,
                "std_macro_f1": item.std_macro_f1,
                "n_folds": len(item.fold_scores),
                "folds": [
                    {
                        "fold": score.fold,
                        "macro_f1": score.macro_f1,
                        "n_labels_evaluated": score.n_labels_evaluated,
                        "n_labels_total": score.n_labels_total,
                        "n_validation_samples": score.n_validation_samples,
                    }
                    for score in item.fold_scores
                ],
                "excluded_labels": list(item.excluded_labels),
                "converged": item.converged,
                "error": item.error,
            }
        )
    return rows


def format_table(rows: Sequence[Mapping]) -> str:
    """Plain-text table of the comparison. A table, never a ranking."""
    header = (
        "strategy".ljust(18)
        + "model".ljust(20)
        + "n_feat".rjust(7)
        + "macro-F1".rjust(10)
        + "std".rjust(8)
        + "folds".rjust(7)
        + "  converged"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        mean = row["mean_macro_f1"]
        spread = row["std_macro_f1"]
        mean_text = "-" if mean is None else f"{mean:.4f}"
        spread_text = "-" if spread is None else f"{spread:.4f}"
        lines.append(
            str(row["strategy"]).ljust(18)
            + str(row["model_id"]).ljust(20)
            + str(row["n_features"]).rjust(7)
            + mean_text.rjust(10)
            + spread_text.rjust(8)
            + str(row["n_folds"]).rjust(7)
            + "  "
            + str(row["converged"])
        )
    return chr(10).join(lines)


def write_json(path: str | Path, payload: Mapping) -> Path:
    """Write deterministic JSON (sorted keys) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return target


class ResultStore:
    """One JSON file with the header, the selections and the finished rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, fingerprint: str) -> dict:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            print(
                f"Note: {self.path} belongs to different data or settings; "
                "starting a new run instead of reusing it."
            )
            return {}
        return payload

    def save(self, payload: Mapping) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> Path:
    """Write the comparison table as CSV (one row per strategy and model)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "strategy",
        "model_id",
        "n_features",
        "mean_macro_f1",
        "std_macro_f1",
        "n_folds",
        "converged",
        "error",
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def subset_for(
    dataset: ProteinDataset,
    plan: SplitPlan,
    settings: ExperimentSettings,
    strategy: str,
    selections: Mapping[str, Mapping],
) -> tuple[str, ...]:
    """Feature subset of one strategy, taken from the development rows only."""
    if strategy == STRATEGY_ALL:
        return tuple(dataset.feature_names)
    if strategy == STRATEGY_VARIANCE:
        return eligible_features(dataset, plan, config=settings.evaluation)
    stored = selections.get(strategy)
    if stored is None:
        raise ValueError(f"Selection for {strategy!r} is missing.")
    return tuple(stored["selected"])


def run_experiment(
    dataset: ProteinDataset,
    plan: SplitPlan,
    settings: ExperimentSettings | None = None,
    *,
    out_dir: str | Path | None = None,
    resume: bool = True,
    sources: Mapping | None = None,
) -> dict:
    """Select features and compare models on the development rows only.

    Args:
        dataset: Aligned features and labels (the full dataset; the test rows are
            copied out of the working data before anything is selected).
        plan: The shared grouped split plan.
        settings: What to compare; defaults to one model per family and all four
            strategies.
        out_dir: Directory for ``results.json`` and ``results.csv``. Nothing is
            written when it is ``None``.
        resume: Reuse stored rows, only when the data fingerprint matches.
        sources: Free-form provenance (input files, seed) recorded in the header.

    Returns:
        The result payload: header, per-strategy selection audit, rows, table.
    """
    config = settings or ExperimentSettings()
    validate_dataset(dataset)
    development, dev_plan = development_block(dataset, plan)
    fingerprint = data_fingerprint(dataset, config)

    store = ResultStore(Path(out_dir) / "results.json") if out_dir else None
    stored = store.load(fingerprint) if (store is not None and resume) else {}
    selections: dict[str, dict] = dict(stored.get("selections") or {})
    evaluations: dict[str, dict] = dict(stored.get("evaluations") or {})

    def payload() -> dict:
        return {
            "storage_version": STORAGE_VERSION,
            "fingerprint": fingerprint,
            "settings": config.as_dict(),
            "sources": dict(sources or {}),
            "data": {
                "n_samples": dataset.n_samples,
                "n_development": int(development.n_samples),
                "n_test": int(np.asarray(plan.test).size),
                "n_features": dataset.n_features,
                "n_labels": dataset.n_labels,
                "feature_names": list(dataset.feature_names),
                "label_names": list(dataset.label_names),
            },
            "selections": selections,
            "evaluations": evaluations,
        }

    for strategy in config.strategies:
        if strategy in evaluations:
            print(f"[{strategy}] reusing the stored comparison")
            continue
        if strategy in (STRATEGY_FORWARD, STRATEGY_BACKWARD) and strategy not in selections:
            started = time.perf_counter()
            selections[strategy] = run_selection(development, dev_plan, config, strategy)
            selections[strategy]["seconds"] = round(time.perf_counter() - started, 3)
            if store is not None:
                store.save(payload())
        features = subset_for(development, dev_plan, config, strategy, selections)
        if not features:
            print(f"[{strategy}] no feature is eligible; skipping its evaluations")
            continue
        print(f"[{strategy}] scoring {len(features)} feature(s) with {len(config.models)} model(s)")
        started = time.perf_counter()
        comparison = compare_models(
            development,
            dev_plan,
            subsets={strategy: features},
            models=config.models,
            config=config.evaluation,
        )
        evaluations[strategy] = {
            "rows": evaluation_rows(comparison),
            "seconds": round(time.perf_counter() - started, 3),
        }
        if store is not None:
            store.save(payload())

    rows = [row for entry in evaluations.values() for row in entry["rows"]]
    result = payload()
    result["rows"] = rows
    result["table"] = format_table(rows)
    if out_dir is not None:
        result["files"] = [str(store.path), str(write_csv(Path(out_dir) / "results.csv", rows))]
        store.save(payload())
    return result


def evaluate_external(
    dataset: ProteinDataset,
    plan: SplitPlan,
    features: Sequence[str],
    model=None,
    *,
    config: ForwardSelectionConfig | None = None,
) -> dict:
    """Score one chosen configuration on the external test block, once.

    The estimator is fitted on every development row and scored on ``plan.test``;
    the labels used are those with enough positives and negatives in the
    development block and at least one positive in the test block. This is the
    only function in the package that indexes the external block, and it is
    never called by :func:`run_experiment`.

    Raises:
        ValueError: If the plan has no test row, the subset is empty or unknown,
            or no label is evaluable on the test block.
    """
    settings = config or ForwardSelectionConfig()
    validate_dataset(dataset)
    names = tuple(features)
    if not names:
        raise ValueError("features is empty.")
    columns = [dataset.feature_names.index(name) for name in names]

    development = np.sort(np.asarray(plan.dev, dtype=int))
    test = np.sort(np.asarray(plan.test, dtype=int))
    if test.size == 0:
        raise ValueError("The split plan has no external test row.")

    y_development = dataset.y[development]
    y_test = dataset.y[test]
    labels_fit = fit_label_mask(
        y_development,
        min_positives=settings.min_positives_fit,
        min_negatives=settings.min_negatives_fit,
    )
    evaluated = evaluable_label_mask(
        y_development,
        y_test,
        min_positives_fit=settings.min_positives_fit,
        min_negatives_fit=settings.min_negatives_fit,
        min_positives_eval=settings.min_positives_eval,
    )
    if not evaluated.any():
        raise ValueError("No label is evaluable on the external test block.")
    fitted_positions = np.flatnonzero(labels_fit)
    evaluated_positions = np.flatnonzero(evaluated)
    positions = [
        int(np.searchsorted(fitted_positions, index)) for index in evaluated_positions
    ]

    estimator = build_estimator(model or settings.estimator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(
            dataset.X[development][:, columns], y_development[:, labels_fit]
        )
        predictions = estimator.predict(dataset.X[test][:, columns])
    metrics = multilabel_metrics(y_test[:, evaluated_positions], predictions[:, positions])
    excluded = [
        name
        for position, name in enumerate(dataset.label_names)
        if not evaluated[position]
    ]
    return {
        "features": list(names),
        "n_features": len(names),
        "model": describe_estimator(model or settings.estimator),
        "n_fit_samples": int(development.size),
        "n_test_samples": int(test.size),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "n_labels_evaluated": metrics["n_labels_evaluated"],
        "n_labels_total": metrics["n_labels_total"],
        "excluded_labels": excluded,
        "converged": not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }
