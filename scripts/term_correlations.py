#!/usr/bin/env python3
"""Correlate the most frequent keyword terms across proteins.

Builds a term x term presence matrix for the top-N ``words`` and ``phrases``
in ``protein_keyword_terms`` and writes two correlation matrices per kind:

* Phi coefficient (Pearson correlation of the binary presence vectors):
  +1 = always together, -1 = mutually exclusive, 0 = independent.
* Jaccard similarity: |A n B| / |A u B|, a pure overlap measure that ignores
  how common each term is on its own.

Usage:
    uv run python scripts/term_correlations.py --top 50
    uv run python scripts/term_correlations.py --kind phrases --top 30
    uv run python scripts/term_correlations.py --out-dir exports/corr
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect

DEFAULT_DB = PROJECT_ROOT / "proteins.duckdb"
DEFAULT_OUT = PROJECT_ROOT / "exports" / "correlations"

TABLE = "protein_keyword_terms"
COLUMNS = {"words": "words", "phrases": "phrases"}


# ---------------------------------------------------------------------------
# Pure math (DB-independent, unit-tested)
# ---------------------------------------------------------------------------


def cooccurrence_matrix(presence: np.ndarray) -> np.ndarray:
    """Pairwise counts of proteins where both terms are present."""
    matrix = presence.astype(np.int32)
    return matrix.T @ matrix


def phi_matrix(presence: np.ndarray) -> np.ndarray:
    """Phi coefficient matrix from a boolean (proteins x terms) array.

    phi = (n11*N - n1*n2) / sqrt(n1*(N-n1)*n2*(N-n2))

    Terms that are always or never present have zero variance, so phi is
    undefined; those entries are reported as 0. The diagonal is 1.
    """
    n = presence.shape[0]
    co = cooccurrence_matrix(presence).astype(np.float64)
    df = presence.sum(axis=0).astype(np.float64)
    expected = np.outer(df, df)
    denominator = np.sqrt(
        np.outer(df, df) * np.outer(n - df, n - df)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (co * n - expected) / denominator
    phi[~np.isfinite(phi)] = 0.0
    np.fill_diagonal(phi, 1.0)
    return phi


def jaccard_matrix(presence: np.ndarray) -> np.ndarray:
    """Jaccard similarity matrix from a boolean (proteins x terms) array."""
    co = cooccurrence_matrix(presence).astype(np.float64)
    df = presence.sum(axis=0).astype(np.float64)
    union = df[:, None] + df[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = co / union
    jaccard[~np.isfinite(jaccard)] = 0.0
    return jaccard


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def top_terms(conn, kind: str, limit: int) -> list[str]:
    """Return the ``limit`` most frequent terms of one kind, deterministic."""
    column = COLUMNS[kind]
    rows = conn.execute(
        "SELECT term, count(*) AS df FROM ("
        f"SELECT unnest(string_split({column}, ' ')) AS term FROM {TABLE}"
        ") WHERE term <> '' "
        "GROUP BY term ORDER BY df DESC, term LIMIT ?",
        [limit],
    ).fetchall()
    return [term for term, _ in rows]


def presence_matrix(conn, kind: str, terms: list[str]) -> np.ndarray:
    """Return a boolean (proteins x terms) presence array."""
    column = COLUMNS[kind]
    select = ", ".join(
        f"list_contains(string_split(COALESCE({column}, ''), ' '), ?) AS t{i}"
        for i in range(len(terms))
    )
    columns = conn.execute(f"SELECT {select} FROM {TABLE}", terms).fetchnumpy()
    return np.column_stack([columns[f"t{i}"] for i in range(len(terms))])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_matrix_csv(path: Path, terms: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term", *terms])
        for term, row in zip(terms, values):
            writer.writerow([term, *(f"{value:.4f}" for value in row)])


def top_pairs(
    terms: list[str], values: np.ndarray, positive: int = 10, negative: int = 5
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Return the strongest positive and negative off-diagonal pairs."""
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            pairs.append((terms[i], terms[j], float(values[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:positive], pairs[::-1][:negative]


def report(kind: str, terms: list[str], phi: np.ndarray, out_dir: Path) -> None:
    write_matrix_csv(out_dir / f"{kind}_phi.csv", terms, phi)
    print(f"\n{kind}: {len(terms)} terms")
    print(f"  matrix -> {out_dir / f'{kind}_phi.csv'}")

    best, worst = top_pairs(terms, phi)
    print(f"  strongest positive phi ({kind}):")
    for a, b, value in best:
        print(f"    {value:+.3f}  {a}  <->  {b}")
    print(f"  strongest negative phi ({kind}):")
    for a, b, value in worst:
        print(f"    {value:+.3f}  {a}  <->  {b}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation matrices for the top keyword terms."
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
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    kinds = ["words", "phrases"] if args.kind == "both" else [args.kind]
    conn = connect(args.db)
    try:
        for kind in kinds:
            terms = top_terms(conn, kind, args.top)
            presence = presence_matrix(conn, kind, terms)
            write_matrix_csv(
                args.out_dir / f"{kind}_jaccard.csv", terms, jaccard_matrix(presence)
            )
            report(kind, terms, phi_matrix(presence), args.out_dir)
            print(f"  matrix -> {args.out_dir / f'{kind}_jaccard.csv'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
