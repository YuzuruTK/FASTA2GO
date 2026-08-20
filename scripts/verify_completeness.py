#!/usr/bin/env python3
"""Verify data completeness in the protein feature database.

Reports how many proteins have all fields populated and shows which columns
(and how many proteins) are missing (NULL) across the analysis data.

Usage:
    uv run python scripts/verify_completeness.py [--db proteins.duckdb]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from protein_database.database.connection import connect
from protein_database.database.models import ANALYSIS_VIEW_SQL

# Columns in `analysis_data` grouped by data category (view column names).
CORE_COLUMNS = [
    "molecular_weight",
    "isoelectric_point",
    "gravy_index",
    "instability_index",
    "aliphatic_index",
    "aromaticity",
    "charge_positive_percentage",
    "charge_negative_percentage",
    "hydrophobic_percentage",
    "polar_percentage",
    "charged_percentage",
    "sulfur_percentage",
    "aromatic_percentage",
    "sequence_entropy",
    "amino_acid_diversity",
    "alanine_percentage",
    "arginine_percentage",
    "asparagine_percentage",
    "aspartic_acid_percentage",
    "cysteine_percentage",
    "glutamine_percentage",
    "glutamic_acid_percentage",
    "glycine_percentage",
    "histidine_percentage",
    "isoleucine_percentage",
    "leucine_percentage",
    "lysine_percentage",
    "methionine_percentage",
    "phenylalanine_percentage",
    "proline_percentage",
    "serine_percentage",
    "threonine_percentage",
    "tryptophan_percentage",
    "tyrosine_percentage",
    "valine_percentage",
]

VALIDATION_COLUMNS = [
    "sequence_quality_score",
    "stability_score",
    "solubility_score",
    "structure_score",
    "overall_score",
    "classification",
]

ANNOTATION_COLUMNS = [
    "uniprot_accession",
    "uniprot_id",
    "protein_name",
    "gene_name",
    "function",
    "subcellular_location",
    "protein_existence",
    "organism",
    "keywords",
    "annotation_status",
    "retrieved_at",
]


def _null_count_sql() -> str:
    """Build a SELECT that returns null counts for every analysis column.

    Returns:
        SQL string yielding rows of (column_name, null_count).
    """
    # Always defined columns that should never be NULL are excluded.
    always_present = {"fasta_id", "source_file", "sequence", "sequence_length", "created_at"}
    all_cols = CORE_COLUMNS + VALIDATION_COLUMNS + ANNOTATION_COLUMNS

    expressions = []
    for col in all_cols:
        if col in always_present:
            continue
        expressions.append(f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}")
    select = ", ".join(expressions)
    return f"""
        SELECT {select}
        FROM analysis_data
    """


def _ensure_view(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the analysis view if it does not exist.

    Uses CREATE OR REPLACE so it is safe to run on any database. This does not
    touch the underlying tables/constraints, so it also works on legacy
    databases that predate the current schema.
    """
    conn.execute(ANALYSIS_VIEW_SQL)


def run_verify(conn, db_path: Path) -> None:
    """Print a data completeness report for the database."""
    # Make sure the analysis view exists (it may predate DB creation).
    _ensure_view(conn)

    total_proteins = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
    print("=" * 60)
    print("  Data Completeness Verification")
    print("=" * 60)
    print(f"  Database         : {db_path}")
    print(f"  Total proteins   : {total_proteins:,}")

    if total_proteins == 0:
        print("  Database is empty.")
        return

    # --- Fully complete (strict: every field non-null) ---
    strict_cols = CORE_COLUMNS + VALIDATION_COLUMNS + ANNOTATION_COLUMNS
    strict_cond = " AND ".join(f"{c} IS NOT NULL" for c in strict_cols)
    strict_count = conn.execute(
        f"SELECT count(*) FROM analysis_data WHERE {strict_cond}"
    ).fetchone()[0]
    print(f"  Fully complete (all fields) : {strict_count:,} ({100 * strict_count / total_proteins:.1f}%)")

    # --- Core complete (computed features + composition only) ---
    core_cond = " AND ".join(f"{c} IS NOT NULL" for c in CORE_COLUMNS)
    core_count = conn.execute(
        f"SELECT count(*) FROM analysis_data WHERE {core_cond}"
    ).fetchone()[0]
    print(f"  Core complete (features/comp): {core_count:,} ({100 * core_count / total_proteins:.1f}%)")

    # --- Per-category presence (whether any field in the category is non-null) ---
    categories = {
        "sequence_features": CORE_COLUMNS[:15],
        "amino_acid_composition": CORE_COLUMNS[15:],
        "validation_features": VALIDATION_COLUMNS,
        "uniprot_annotations": ANNOTATION_COLUMNS,
    }
    print("\n  Proteins with data per category (any field populated):")
    for label, cols in categories.items():
        present_cond = " OR ".join(f"{c} IS NOT NULL" for c in cols)
        count = conn.execute(
            f"SELECT count(*) FROM analysis_data WHERE {present_cond}"
        ).fetchone()[0]
        print(f"    {label:<24}: {count:>10,} ({100 * count / total_proteins:.1f}%)")

    # --- Per-column null counts ---
    print("\n  Per-column NULL counts:")
    print(f"    {'Column':<30}{'NULLs':>12}{'% missing':>12}")
    print("    " + "-" * 54)
    row = conn.execute(_null_count_sql()).fetchone()
    col_names = [
        c for c in CORE_COLUMNS + VALIDATION_COLUMNS + ANNOTATION_COLUMNS
        if c not in {"fasta_id", "source_file", "sequence", "sequence_length", "created_at"}
    ]
    for name, value in zip(col_names, row):
        pct = 100 * value / total_proteins if value else 0.0
        print(f"    {name:<30}{value:>12,}{pct:>11.1f}%")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify data completeness in the protein feature database."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="proteins.duckdb",
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