#!/usr/bin/env python3
"""CLI entry point for UniProt annotation enrichment.

Usage:
    uv run python scripts/annotate_proteins.py --db proteins.duckdb
    uv run python scripts/annotate_proteins.py --db proteins.duckdb --source cache
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.uniprot import annotate_database

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich the protein database with UniProt annotations."
    )
    parser.add_argument(
        "--db", type=str, default="proteins.duckdb",
        help="Path to the DuckDB database (default: proteins.duckdb).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache/uniprot",
        help="Directory for the snapshot and any legacy TSVs "
        "(default: cache/uniprot).",
    )
    parser.add_argument(
        "--source", type=str, choices=["bulk", "cache"], default="bulk",
        help="'bulk' downloads the reviewed-proteome snapshot; 'cache' replays "
        "local TSV files only. Default: bulk.",
    )
    parser.add_argument(
        "--refresh-snapshot", action="store_true",
        help="Re-download the bulk snapshot even when cached.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  UniProt Annotation Enrichment")
    print("=" * 60)
    print(f"  Database : {db_path}")
    print(f"  Source   : {args.source}")
    print(f"  Cache    : {args.cache_dir}")
    print("-" * 60)

    try:
        summary = annotate_database(
            db_path=db_path,
            cache_dir=args.cache_dir,
            source=args.source,
            refresh_snapshot=args.refresh_snapshot,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  UniProt Annotation Summary")
    print("=" * 60)
    print(f"  {'Total local proteins':<22}: {summary['total_local_proteins']:,}")
    print(f"  {'Annotated rows':<22}: {summary['annotated']:,}")
    print(f"  {'Missing Function':<22}: {summary['missing_function']:,}")
    print(f"  {'Not found':<22}: {summary['not_found']:,}")
    print(f"  {'Function coverage':<22}: {summary['function_coverage_pct']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
