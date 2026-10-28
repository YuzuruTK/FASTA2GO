#!/usr/bin/env python3
"""Run the feature-selection x model comparison on the development rows.

Usage:
    # inspect the data and the split plan without fitting anything
    uv run python scripts/run_experiment.py --limit 50 --dry-run

    # run the comparison (development rows only)
    uv run python scripts/run_experiment.py --out exports/experiments/sample

    # score one chosen configuration on the external test block, once
    uv run python scripts/run_experiment.py --out exports/experiments/sample \
        --evaluate-test --test-strategy all_features --test-model logistic_ovr

Everything that selects or tunes runs on the development rows. The external
test block is scored only when --evaluate-test is given, and only for the
configuration named on the command line: this script never picks a winner.
"""

import argparse
import csv
import sys
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.ml.dataset import (
    align,
    load_features,
    load_labels_csv,
    load_sequence_groups,
)
from protein_database.ml.experiments import (
    ExperimentSettings,
    evaluate_external,
    run_experiment,
    write_json,
)
from protein_database.ml.feature_selection import ForwardSelectionConfig
from protein_database.ml.models import (
    LogisticConfig,
    RandomForestConfig,
    XGBoostConfig,
    describe_estimator,
)
from protein_database.ml.splits import make_group_split_plan, validate_split_plan

DEFAULT_LABELS = "exports/go_slim/sample-5000/protein_labels.csv"
DEFAULT_VOCABULARY = "data/go_slim/vocabulary_mf.csv"
DEFAULT_STRATEGIES = "all_features,variance_prefilter,forward,backward"
DEFAULT_MODELS = "logistic,forest,xgboost"

MODELS = {
    "logistic": LogisticConfig,
    "forest": RandomForestConfig,
    "xgboost": XGBoostConfig,
}


def read_vocabulary(path: str | Path) -> tuple[str, ...]:
    """GO identifiers of the frozen Molecular Function vocabulary, in file order."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "go_id" not in reader.fieldnames:
            raise ValueError(
                f"{source} has no go_id column (found {reader.fieldnames})."
            )
        names = tuple(row["go_id"].strip() for row in reader if row.get("go_id"))
    if not names:
        raise ValueError(f"{source} contains no GO term.")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare feature-selection strategies and models on the "
        "development rows of the labelled protein sample."
    )
    parser.add_argument("--db", default="proteins.duckdb", help="DuckDB database.")
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help=f"Label CSV (default: {DEFAULT_LABELS}).",
    )
    parser.add_argument(
        "--vocabulary",
        default=DEFAULT_VOCABULARY,
        help=f"Frozen GO vocabulary (default: {DEFAULT_VOCABULARY}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for results.json and results.csv (default: write nothing).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N accessions."
    )
    parser.add_argument(
        "--n-folds", type=int, default=3, help="Inner grouped folds (default: 3)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of groups held out for testing (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42).")
    parser.add_argument(
        "--strategies",
        default=DEFAULT_STRATEGIES,
        help=f"Comma-separated strategies (default: {DEFAULT_STRATEGIES}).",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"Comma-separated model families (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10,
        help="Forward/backward feature ceiling (default: 10).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-4,
        help="Minimum gain to accept a step (default: 1e-4).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore stored results and recompute every strategy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the data and the plan, then stop without fitting.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Score one chosen configuration on the external test block, once.",
    )
    parser.add_argument(
        "--test-strategy", default=None, help="Strategy of the configuration to test."
    )
    parser.add_argument(
        "--test-model",
        default=None,
        help="Model id to test (logistic_ovr, random_forest_ovr, xgboost_ovr).",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> ExperimentSettings:
    families = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in families if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model family/families {unknown}; use {list(MODELS)}.")
    selection = dict(
        max_features=args.max_features, min_improvement=args.min_improvement
    )
    return ExperimentSettings(
        models=tuple(MODELS[name]() for name in families),
        strategies=tuple(
            name.strip() for name in args.strategies.split(",") if name.strip()
        ),
        forward=ForwardSelectionConfig(**selection),
        backward=ForwardSelectionConfig(**selection),
        seed=args.seed,
    )


def choose_subset(args, result, dataset) -> tuple[str, ...]:
    """Feature subset of the strategy named on the command line."""
    if args.test_strategy in ("forward", "backward"):
        audit = result["selections"].get(args.test_strategy)
        if audit is None:
            raise SystemExit(f"Strategy {args.test_strategy!r} was not run.")
        return tuple(audit["selected"])
    if args.test_strategy == "all_features":
        return tuple(dataset.feature_names)
    rows = [
        row for row in result["rows"] if row["strategy"] == args.test_strategy
    ]
    if not rows:
        raise SystemExit(f"Strategy {args.test_strategy!r} produced no subset.")
    return tuple(rows[0]["features"])


def main() -> None:
    args = parse_args()
    if args.evaluate_test and not (args.test_strategy and args.test_model):
        raise SystemExit(
            "--evaluate-test needs --test-strategy and --test-model: the "
            "configuration is chosen on the development rows, not by the script."
        )

    label_names = read_vocabulary(args.vocabulary)
    labels = load_labels_csv(args.labels, label_names)
    accessions = sorted(labels.labels)
    if args.limit is not None:
        accessions = accessions[: args.limit]

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        features = load_features(conn, accessions)
        groups = load_sequence_groups(conn, accessions)
    finally:
        conn.close()

    dataset = align(features, labels)
    row_groups = np.array([groups[key.accession] for key in dataset.keys])
    plan = make_group_split_plan(
        row_groups, n_folds=args.n_folds, test_size=args.test_size, seed=args.seed
    )
    validate_split_plan(plan, row_groups, n_samples=dataset.n_samples)

    print("=" * 68)
    print("  FASTA2GO experiment (development rows only)")
    print("=" * 68)
    print(f"  Database      : {db_path}")
    print(f"  Labels        : {args.labels}")
    print(f"  Vocabulary    : {args.vocabulary} ({len(label_names)} terms)")
    print(f"  Proteins      : {dataset.n_samples:,} rows in the labelled sample")
    print(f"  Features      : {dataset.n_features}")
    print(
        f"  Groups        : {len(set(row_groups.tolist())):,} distinct md5(sequence) "
        "values (identical sequences only)"
    )
    print(
        f"  Split         : {plan.n_folds} folds, {int(plan.dev.size):,} development "
        f"rows, {int(plan.test.size):,} external test rows"
    )
    print("=" * 68)

    if args.dry_run:
        print("Dry run: nothing was fitted and nothing was written.")
        return

    settings = build_settings(args)
    result = run_experiment(
        dataset,
        plan,
        settings,
        out_dir=args.out,
        resume=not args.no_resume,
        sources={
            "db": str(db_path),
            "labels": str(args.labels),
            "vocabulary": str(args.vocabulary),
            "seed": settings.seed,
            "accessions": len(accessions),
        },
    )

    for strategy in ("forward", "backward"):
        audit = result["selections"].get(strategy)
        if audit is None:
            continue
        selected = audit["n_selected"]
        stopped = audit["stopped_reason"]
        print(f"[{strategy}] {selected} feature(s) selected, stopped on {stopped}")
    print()
    print(result["table"])
    print()
    if args.out:
        files = ", ".join(result["files"])
        print(f"Written: {files}")
    print(
        "This table is a comparison, not a ranking: the configuration is chosen on "
        "these development numbers, and the external test block has not been read."
    )

    if not args.evaluate_test:
        return

    subset = choose_subset(args, result, dataset)
    chosen = [
        model
        for model in settings.models
        if describe_estimator(model)["model_id"] == args.test_model
    ]
    if not chosen:
        raise SystemExit(f"Model {args.test_model!r} is not part of this run.")

    report = evaluate_external(
        dataset, plan, subset, chosen[0], config=settings.evaluation
    )
    n_features = report["n_features"]
    model_id = report["model"]["model_id"]
    n_fit = report["n_fit_samples"]
    n_test = report["n_test_samples"]
    macro = report["macro_f1"]
    micro = report["micro_f1"]
    n_evaluated = report["n_labels_evaluated"]
    n_total = report["n_labels_total"]
    print()
    print("External test block (evaluated once)")
    print(f"  Strategy       : {args.test_strategy} ({n_features} features)")
    print(f"  Model          : {model_id}")
    print(f"  Fit rows       : {n_fit:,} development")
    print(f"  Test rows      : {n_test:,}")
    print(f"  Macro-F1       : {macro:.4f}")
    print(f"  Micro-F1       : {micro:.4f}")
    print(f"  Labels scored  : {n_evaluated} of {n_total}")
    excluded = report["excluded_labels"]
    if excluded:
        print(f"  Excluded labels: {len(excluded)}")
    if args.out:
        path = write_json(Path(args.out) / "external_test.json", report)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
