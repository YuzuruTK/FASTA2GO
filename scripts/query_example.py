#!/usr/bin/env python3
"""Example script to query the protein database and display results.

Usage:
    uv run python scripts/query_example.py [--db proteins.duckdb]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the protein feature database."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="proteins.duckdb",
        help="Path to the DuckDB database (default: proteins.duckdb).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to display (default: 10).",
    )
    parser.add_argument(
        "--longest",
        action="store_true",
        help="Show the longest protein sequences.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show database summary statistics.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)

    try:
        if args.summary:
            show_summary(conn)
        elif args.longest:
            show_longest(conn, args.top)
        else:
            show_recent(conn, args.top)
    finally:
        conn.close()


def show_summary(conn) -> None:
    """Display summary statistics about the database."""
    total_proteins = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]

    if total_proteins == 0:
        print("Database is empty.")
        return

    total_features = conn.execute("SELECT count(*) FROM sequence_features").fetchone()[0]
    total_compositions = conn.execute("SELECT count(*) FROM amino_acid_composition").fetchone()[0]

    row = conn.execute(
        """
        SELECT
            avg(sequence_length),
            min(sequence_length),
            max(sequence_length),
            count(DISTINCT source_file)
        FROM proteins
        """
    ).fetchone()
    avg_length, min_length, max_length, distinct_files = row

    print("=" * 60)
    print("  Database Summary")
    print("=" * 60)
    print(f"  Total proteins        : {total_proteins:,}")
    print(f"  Sequence features     : {total_features:,}")
    print(f"  Amino acid comp.      : {total_compositions:,}")
    print(f"  Source files          : {distinct_files}")
    print(f"  Avg sequence length   : {avg_length:.1f}")
    print(f"  Min sequence length   : {min_length}")
    print(f"  Max sequence length   : {max_length}")
    print("=" * 60)

    # Show top 5 source files
    print("\n  Source files:")
    results = conn.execute(
        """
        SELECT source_file, count(*) AS n
        FROM proteins
        GROUP BY source_file
        ORDER BY n DESC
        LIMIT 5
        """
    ).fetchall()
    for file_name, count in results:
        print(f"    {file_name}: {count:,} proteins")


def show_longest(conn, top_n: int) -> None:
    """Display the longest protein sequences."""
    print(f"Top {top_n} Longest Protein Sequences:")
    print("=" * 80)
    results = conn.execute(
        """
        SELECT fasta_id, description, sequence_length, source_file
        FROM proteins
        ORDER BY sequence_length DESC
        LIMIT ?
        """,
        [top_n],
    ).fetchall()
    for i, (fasta_id, description, seq_len, source_file) in enumerate(results, 1):
        desc = (description or "")[:60]
        print(f"  {i:3d}. {fasta_id}")
        print(f"       Description: {desc}")
        print(f"       Length     : {seq_len:,} aa")
        print(f"       Source     : {source_file}")
        print()


def show_recent(conn, top_n: int) -> None:
    """Display the most recently added proteins with their features."""
    print(f"Latest {top_n} Proteins (with features):")
    print("=" * 100)
    results = conn.execute(
        """
        SELECT
            p.fasta_id,
            p.description,
            p.sequence_length,
            sf.molecular_weight,
            sf.isoelectric_point,
            sf.gravy_index,
            sf.instability_index
        FROM proteins p
        LEFT JOIN sequence_features sf
            ON p.fasta_id = sf.fasta_id AND p.source_file = sf.source_file
        ORDER BY p.created_at DESC, p.fasta_id DESC
        LIMIT ?
        """,
        [top_n],
    ).fetchall()
    for fasta_id, description, seq_len, mw, iso, gravy, instab in results:
        print(f"  FASTA: {fasta_id}")
        print(f"  Desc : {description[:80] if description else 'N/A'}")
        print(f"  Len  : {seq_len:,} aa")
        if mw is not None:
            print(f"  MW   : {mw:.2f} Da")
            print(f"  pI   : {iso:.2f}")
            print(f"  GRAVY: {gravy:.3f}")
            print(f"  Instab: {instab:.2f}")
        print()


if __name__ == "__main__":
    main()