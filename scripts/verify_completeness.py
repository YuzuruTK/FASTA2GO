#!/usr/bin/env python3
"""Report data completeness in the protein feature database.

Usage:
    uv run python scripts/verify_completeness.py [--db proteins.duckdb]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect
from protein_database.schema import (
    ANALYSIS_VIEW_SQL,
    COMPOSITION_COLUMNS,
    FEATURE_COLUMNS,
    VIEW_ANNOTATION_COLUMNS,
)

CORE_COLUMNS = FEATURE_COLUMNS + COMPOSITION_COLUMNS

#: Columns in `analysis_data` grouped by data category.
CATEGORIES = {
    "sequence_features": FEATURE_COLUMNS,
    "amino_acid_composition": COMPOSITION_COLUMNS,
    "uniprot_annotations": VIEW_ANNOTATION_COLUMNS + ["annotation_status"],
}


def run_verify(conn, db_path: Path) -> None:
    """Print a data completeness report for the database."""
    conn.execute(ANALYSIS_VIEW_SQL)  # safe on legacy databases

    total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
    print("=" * 60)
    print("  Data Completeness Verification")
    print("=" * 60)
    print(f"  Database       : {db_path}")
    print(f"  Total proteins : {total:,}")
    if total == 0:
        print("  Database is empty.")
        return

    # Fully complete: every core and annotation field non-null.
    all_columns = CORE_COLUMNS + CATEGORIES["uniprot_annotations"]
    complete = conn.execute(
        "SELECT count(*) FROM analysis_data WHERE "
        + " AND ".join(f"{column} IS NOT NULL" for column in all_columns)
    ).fetchone()[0]
    print(f"  Fully complete : {complete:,} ({100 * complete / total:.1f}%)")

    print("\n  Proteins with data per category (any field populated):")
    for label, columns in CATEGORIES.items():
        present = conn.execute(
            "SELECT count(*) FROM analysis_data WHERE "
            + " OR ".join(f"{column} IS NOT NULL" for column in columns)
        ).fetchone()[0]
        print(f"    {label:<24}: {present:>10,} ({100 * present / total:.1f}%)")

    print("\n  Per-column NULL counts:")
    print(f"    {'Column':<30}{'NULLs':>12}{'% missing':>12}")
    print("    " + "-" * 54)
    row = conn.execute(
        "SELECT "
        + ", ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END)"
            for column in all_columns
        )
        + " FROM analysis_data"
    ).fetchone()
    for name, value in zip(all_columns, row):
        pct = 100 * value / total if value else 0.0
        print(f"    {name:<30}{value:>12,}{pct:>11.1f}%")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify data completeness in the protein feature database."
    )
    parser.add_argument(
        "--db", type=str, default="proteins.duckdb",
        help="Path to the DuckDB database (default: proteins.duckdb).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)
    try:
        run_verify(conn, db_path)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
