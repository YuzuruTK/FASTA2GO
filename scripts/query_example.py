#!/usr/bin/env python3
"""Example script to query the protein database and display results.

Usage:
    uv run python scripts/query_example.py [--db proteins.db]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from protein_database.database.models import (
    Protein,
    SequenceFeature,
    AminoAcidComposition,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the protein feature database."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="proteins.db",
        help="Path to the SQLite database (default: proteins.db).",
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

    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    if args.summary:
        show_summary(engine)
    elif args.longest:
        show_longest(engine, args.top)
    else:
        show_recent(engine, args.top)


def show_summary(engine) -> None:
    """Display summary statistics about the database."""
    with Session(engine) as session:
        total_proteins = session.query(func.count(Protein.id)).scalar()
        total_features = session.query(func.count(SequenceFeature.id)).scalar()
        total_compositions = session.query(func.count(AminoAcidComposition.id)).scalar()

        if total_proteins == 0:
            print("Database is empty.")
            return

        avg_length = session.query(func.avg(Protein.sequence_length)).scalar()
        min_length = session.query(func.min(Protein.sequence_length)).scalar()
        max_length = session.query(func.max(Protein.sequence_length)).scalar()
        distinct_files = session.query(func.count(func.distinct(Protein.source_file))).scalar()

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
        results = (
            session.query(Protein.source_file, func.count(Protein.id))
            .group_by(Protein.source_file)
            .order_by(func.count(Protein.id).desc())
            .limit(5)
            .all()
        )
        for file_name, count in results:
            print(f"    {file_name}: {count:,} proteins")


def show_longest(engine, top_n: int) -> None:
    """Display the longest protein sequences."""
    with Session(engine) as session:
        results = (
            session.query(Protein)
            .order_by(Protein.sequence_length.desc())
            .limit(top_n)
            .all()
        )

        print(f"Top {top_n} Longest Protein Sequences:")
        print("=" * 80)
        for i, protein in enumerate(results, 1):
            description = (protein.description or "")[:60]
            print(f"  {i:3d}. {protein.fasta_id}")
            print(f"       Description: {description}")
            print(f"       Length     : {protein.sequence_length:,} aa")
            print(f"       Source     : {protein.source_file}")
            print()


def show_recent(engine, top_n: int) -> None:
    """Display the most recently added proteins with their features."""
    with Session(engine) as session:
        results = (
            session.query(Protein, SequenceFeature)
            .outerjoin(SequenceFeature, Protein.id == SequenceFeature.protein_id)
            .order_by(Protein.id.desc())
            .limit(top_n)
            .all()
        )

        print(f"Latest {top_n} Proteins (with features):")
        print("=" * 100)
        for protein, features in results:
            print(f"  ID   : {protein.id}")
            print(f"  FASTA: {protein.fasta_id}")
            print(f"  Desc : {protein.description[:80] if protein.description else 'N/A'}")
            print(f"  Len  : {protein.sequence_length:,} aa")
            if features:
                print(f"  MW   : {features.molecular_weight:.2f} Da")
                print(f"  pI   : {features.isoelectric_point:.2f}")
                print(f"  GRAVY: {features.gravy_index:.3f}")
                print(f"  Instab: {features.instability_index:.2f}")
            print()


if __name__ == "__main__":
    main()