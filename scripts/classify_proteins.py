#!/usr/bin/env python3
"""Classify proteins in the database using the curated functional labels.

Reads ``data/functional_labels.csv`` (produced by
``scripts/build_functional_labels.py``) and scores every protein
description in the ``analysis_data`` view of the DuckDB database.

Scoring model (evidence-based, not exclusive):

* Each matched keyword adds its weight to the corresponding class.
* A protein may match multiple classes (multi-label).
* A class is assigned when its score reaches ``--threshold`` (default 1.0),
  so a lone weak word like "membrane" (0.3) is not enough, while
  "catalyzes" (1.0) or "kinase" + "binding" (1.2) are.

Usage:
    uv run python scripts/classify_proteins.py
    uv run python scripts/classify_proteins.py --threshold 0.5 \
        --export-assignments exports/protein_classes.csv \
        --export-unclassified exports/unclassified.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect
from scripts.lexicon import tokenize

DEFAULT_LABELS = PROJECT_ROOT / "data" / "functional_labels.csv"
DEFAULT_DB = PROJECT_ROOT / "proteins.duckdb"

#: Canonical class order for reporting.
CLASSES = [
    "enzyme",
    "transporter",
    "dna_rna_binding",
    "structural",
    "receptor_signaling",
    "motor",
    "defense",
    "storage",
]


# ---------------------------------------------------------------------------
# Core logic (pure functions, DB-independent)
# ---------------------------------------------------------------------------


def load_labels(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load the functional labels CSV.

    Returns a mapping word -> [(class, weight), ...].
    """
    labels: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"word", "class", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must have columns 'word', 'class' and 'weight'"
            )
        for row in reader:
            word = row["word"].strip().lower()
            cls = row["class"].strip()
            weight = float(row["weight"])
            if word:
                labels[word].append((cls, weight))
    return dict(labels)


def score_description(
    words: list[str], labels: dict[str, list[tuple[str, float]]]
) -> tuple[dict[str, float], list[str]]:
    """Score a tokenized description against the label dictionary.

    Returns (scores, matched_words) where scores maps class -> summed
    weight and matched_words is the deduplicated list of label words hit.
    """
    scores: dict[str, float] = defaultdict(float)
    matched: list[str] = []
    for word in sorted(set(words)):
        for cls, weight in labels.get(word, []):
            scores[cls] += weight
            matched.append(word)
    return dict(scores), matched


def assigned_classes(scores: dict[str, float], threshold: float) -> list[str]:
    """Return classes whose score reaches the threshold, canonically sorted."""
    return sorted(
        (cls for cls, score in scores.items() if score >= threshold),
        key=lambda cls: (CLASSES.index(cls) if cls in CLASSES else 99, cls),
    )


def top_class(scores: dict[str, float]) -> str | None:
    """Return the highest-scoring class (alphabetical tie-break)."""
    if not scores:
        return None
    return min(scores.items(), key=lambda item: (-item[1], item[0]))[0]


# ---------------------------------------------------------------------------
# Reporting / exports
# ---------------------------------------------------------------------------


def write_assignments_csv(rows: list[dict], path: Path) -> None:
    header = ["fasta_id", "source_file", "top_class", "classes", "scores",
              "matched_words"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_unclassified_csv(rows: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fasta_id", "source_file", "function"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many proteins in the database match each functional "
            "class using the curated keyword labels."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to the DuckDB database (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Functional labels CSV (default: {DEFAULT_LABELS}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum class score for a protein to count toward that class "
             "(default: 1.0).",
    )
    parser.add_argument(
        "--export-assignments",
        type=Path,
        default=None,
        help="Optional CSV with per-protein class assignments.",
    )
    parser.add_argument(
        "--export-unclassified",
        type=Path,
        default=None,
        help="Optional CSV with proteins that matched no class.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    if not args.labels.exists():
        print(f"Error: Labels file not found: {args.labels}", file=sys.stderr)
        print(
            "Generate it first with: "
            "uv run python scripts/build_functional_labels.py",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = load_labels(args.labels)

    conn = connect(args.db)
    try:
        rows = conn.execute(
            "SELECT fasta_id, source_file, function FROM analysis_data "
            "WHERE function IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    class_counts: Counter[str] = Counter()
    word_evidence: dict[str, Counter[str]] = defaultdict(Counter)
    classified = 0
    unclassified_rows: list[tuple] = []
    assignment_rows: list[dict] = []

    for fasta_id, source_file, function_text in rows:
        words = tokenize(function_text)
        scores, matched = score_description(words, labels)
        classes = assigned_classes(scores, args.threshold)

        if classes:
            classified += 1
            class_counts.update(classes)
            for word in matched:
                for cls, _ in labels[word]:
                    word_evidence[cls][word] += 1
        else:
            unclassified_rows.append((fasta_id, source_file, function_text))

        if args.export_assignments:
            best = top_class(scores)
            assignment_rows.append(
                {
                    "fasta_id": fasta_id,
                    "source_file": source_file,
                    "top_class": best or "",
                    "classes": ";".join(classes),
                    "scores": ";".join(
                        f"{cls}:{score:.2f}"
                        for cls, score in sorted(scores.items())
                    ),
                    "matched_words": ";".join(matched),
                }
            )

    total = len(rows)
    unclassified = total - classified

    # Summary.
    print("=" * 60)
    print("  Protein classification — curated functional labels")
    print("=" * 60)
    print(f"  Labels loaded          : {len(labels):,} words")
    print(f"  Score threshold        : {args.threshold}")
    print(f"  Proteins with function : {total:,}")
    print(f"  Classified             : {classified:,} "
          f"({classified / total * 100:.1f}%)" if total else "  Classified: 0")
    print(f"  Unclassified           : {unclassified:,} "
          f"({unclassified / total * 100:.1f}%)" if total else "  Unclassified: 0")
    print("=" * 60)

    print("\nBy class (multi-label; a protein may count toward several):")
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        pct = (count / total * 100) if total else 0.0
        print(f"  {cls:<20} {count:>9,}  ({pct:5.1f}%)")

    print("\nTop evidence words per class:")
    for cls in CLASSES:
        evidence = word_evidence.get(cls)
        if not evidence:
            continue
        top = ", ".join(
            f"{word} ({count:,})"
            for word, count in evidence.most_common(10)
        )
        print(f"  {cls}: {top}")

    if args.export_assignments:
        write_assignments_csv(assignment_rows, args.export_assignments)
        print(f"\nExported {len(assignment_rows):,} assignments to: "
              f"{args.export_assignments}")
    if args.export_unclassified:
        write_unclassified_csv(unclassified_rows, args.export_unclassified)
        print(f"Exported {len(unclassified_rows):,} unclassified rows to: "
              f"{args.export_unclassified}")


if __name__ == "__main__":
    main()