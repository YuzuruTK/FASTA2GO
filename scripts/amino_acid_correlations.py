#!/usr/bin/env python3
"""Correlate amino-acid composition with the top keyword terms.

Builds a (20 amino acids) x (top-N terms) matrix of Pearson correlations
between each amino-acid percentage and the presence of each word/phrase.
Because the term side is binary, this is the point-biserial correlation:

    r = 0  no association
    r = +1 every protein carrying the term has the maximum percentage
    r = -1 every protein carrying the term has the minimum percentage

Values near 0 mean the term says nothing about that residue's abundance.
Note that percentages are compositional (each protein's 20 values sum to
100), so the 20 rows are not independent of one another.

Usage:
    uv run python scripts/amino_acid_correlations.py --top 50
    uv run python scripts/amino_acid_correlations.py --kind phrases --top 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect
from protein_database.schema import AMINO_ACIDS, COMPOSITION_COLUMNS
from scripts.term_correlations import top_terms

DEFAULT_DB = PROJECT_ROOT / "proteins.duckdb"
DEFAULT_OUT = PROJECT_ROOT / "exports" / "correlations"

TABLE = "protein_keyword_terms"
COLUMNS = {"words": "words", "phrases": "phrases"}


# ---------------------------------------------------------------------------
# Pure math (DB-independent, unit-tested)
# ---------------------------------------------------------------------------


def correlation_matrix_chunks(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Pearson correlation between columns of x and columns of y.

    ``chunks`` yields ``(x, y)`` row blocks with a fixed column layout, so
    the full table never has to be materialized. Returns a
    ``(x_columns, y_columns)`` matrix; constant columns correlate as 0.
    """
    n = 0
    sx = sy = sx2 = sy2 = None
    sxy = None

    for x, y in chunks:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if sx is None:
            sx = np.zeros(x.shape[1])
            sx2 = np.zeros(x.shape[1])
            sy = np.zeros(y.shape[1])
            sy2 = np.zeros(y.shape[1])
            sxy = np.zeros((x.shape[1], y.shape[1]))
        n += x.shape[0]
        sx += x.sum(axis=0)
        sx2 += (x * x).sum(axis=0)
        sy += y.sum(axis=0)
        sy2 += (y * y).sum(axis=0)
        sxy += x.T @ y

    numerator = n * sxy - np.outer(sx, sy)
    denominator = np.sqrt(
        np.outer(n * sx2 - sx * sx, n * sy2 - sy * sy)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = numerator / denominator
    correlation[~np.isfinite(correlation)] = 0.0
    return np.clip(correlation, -1.0, 1.0)


def correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convenience wrapper for an in-memory (non-chunked) correlation."""
    return correlation_matrix_chunks([(x, y)])


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _presence_sql(term_count: int, column: str) -> str:
    return ", ".join(
        f"list_contains(string_split(COALESCE({column}, ''), ' '), ?) AS t{i}"
        for i in range(term_count)
    )


def correlation_chunks(
    conn, kind: str, terms: list[str], batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (amino-acid percentages, term presence) row blocks."""
    column = COLUMNS[kind]
    amino_acids = ", ".join(f"aac.{name}" for name in COMPOSITION_COLUMNS)
    sql = (
        f"SELECT {amino_acids}, {_presence_sql(len(terms), column)} "
        f"FROM {TABLE} k "
        "JOIN amino_acid_composition aac USING (fasta_id, source_file)"
    )
    cursor = conn.execute(sql, terms)
    split = len(COMPOSITION_COLUMNS)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        block = np.array(rows, dtype=np.float64)
        yield block[:, :split], block[:, split:]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_matrix_csv(path: Path, row_labels: list[str], column_labels: list[str],
                     values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["amino_acid", *column_labels])
        for label, row in zip(row_labels, values):
            writer.writerow([label, *(f"{value:+.4f}" for value in row)])


def strongest(amino_acids: list[str], terms: list[str], values: np.ndarray,
              limit: int = 10) -> list[tuple[str, str, float]]:
    pairs = [
        (amino_acid, term, float(values[i, j]))
        for i, amino_acid in enumerate(amino_acids)
        for j, term in enumerate(terms)
    ]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs[:limit]


def report(kind: str, amino_acids: list[str], terms: list[str],
           values: np.ndarray, path: Path) -> None:
    write_matrix_csv(path, amino_acids, terms, values)
    print(f"\n{kind}: {len(amino_acids)} amino acids x {len(terms)} terms")
    print(f"  matrix -> {path}")
    print(f"  strongest absolute correlations ({kind}):")
    for amino_acid, term, value in strongest(amino_acids, terms, values):
        print(f"    {value:+.3f}  {amino_acid:<14} <->  {term}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate amino-acid composition with keyword terms."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Path to the DuckDB database (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of terms per kind (default: 50).",
    )
    parser.add_argument(
        "--kind", choices=["both", "words", "phrases"], default="both",
        help="Which term kind to correlate (default: both).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help=f"Directory for the matrices (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=25_000,
        help="Rows per streamed batch (default: 25000).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    amino_acids = list(AMINO_ACIDS.values())
    kinds = ["words", "phrases"] if args.kind == "both" else [args.kind]
    conn = connect(args.db)
    try:
        for kind in kinds:
            terms = top_terms(conn, kind, args.top)
            values = correlation_matrix_chunks(
                correlation_chunks(conn, kind, terms, args.batch_size)
            )
            report(
                kind, amino_acids, terms, values,
                args.out_dir / f"amino_acids_vs_{kind}.csv",
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
