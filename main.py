#!/usr/bin/env python3
"""CLI entry point for the Protein Sequence Feature Database.

Usage:
    uv run python main.py --input uniprot_sprot.fasta
    uv run python main.py --input uniprot_sprot.fasta --db proteins.db --batch-size 500 --limit 1000
"""

import argparse
import sys

from protein_database.pipeline import process_fasta_file


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def main() -> None:
    """Parse CLI arguments and run the protein processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Protein Sequence Feature Database - Parse FASTA files, "
        "extract features, and store results in SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input uniprot_sprot.fasta\n"
            "  python main.py --input uniprot_sprot.fasta --db my_proteins.db\n"
            "  python main.py --input test.fasta --limit 100 --batch-size 500\n"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input FASTA file.",
    )

    parser.add_argument(
        "--db", "-d",
        type=str,
        default="proteins.db",
        help="Path to the output SQLite database file (default: proteins.db).",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Number of records to batch-commit to the database (default: 1000).",
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of proteins to process (for testing).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the database before processing.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Protein Sequence Feature Database")
    print("=" * 60)
    print(f"  Input FASTA : {args.input}")
    print(f"  Database     : {args.db}")
    print(f"  Batch size   : {args.batch_size:,}")
    if args.limit:
        print(f"  Limit        : {args.limit:,}")
    if args.overwrite:
        print(f"  Overwrite    : Yes")
    print("-" * 60)

    try:
        summary = process_fasta_file(
            fasta_path=args.input,
            db_path=args.db,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
        )

        print("=" * 60)
        print("  Processing Complete")
        print("=" * 60)
        print(f"  Processed    : {summary['processed']:,} proteins")
        print(f"  Skipped      : {summary['skipped']:,} proteins (already in DB)")
        print(f"  Errors       : {summary['errors']}")
        print(f"  Elapsed time : {format_time(summary['elapsed_seconds'])}")
        print(f"  Database     : {summary['database_path']}")
        print(f"  Rate         : {summary['processed'] / summary['elapsed_seconds']:.1f} proteins/sec")
        print("=" * 60)

    except FileNotFoundError:
        print(f"Error: FASTA file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()