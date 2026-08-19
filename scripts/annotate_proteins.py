#!/usr/bin/env python3
"""CLI entry point for UniProt annotation enrichment.

Usage:
    uv run python scripts/annotate_proteins.py --db proteins.duckdb
    uv run python scripts/annotate_proteins.py --db proteins.duckdb --batch-size 100
    uv run python scripts/annotate_proteins.py --db proteins.duckdb --limit 5000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.uniprot.annotate import annotate_database, RunSummary

logger = logging.getLogger("annotate_proteins")


def format_summary(s: RunSummary) -> str:
    """Format a RunSummary as a human-readable block."""
    lines = [
        "=" * 60,
        "  UniProt Annotation Summary",
        "=" * 60,
        f"  Total local proteins        : {s.total_local_proteins:,}",
        f"  Unique UniProt accessions   : {s.unique_accessions:,}",
        f"  Successfully retrieved      : {s.successfully_retrieved:,}",
        f"  Missing Function            : {s.missing_function:,}",
        f"  Not found                   : {s.not_found:,}",
        f"  Failed                      : {s.failed:,}",
        f"  HTTP requests               : {s.http_requests:,}",
        f"  Cached batches              : {s.cached_batches:,}",
        f"  Function coverage           : {s.function_coverage_pct:.2f}%",
        "=" * 60,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich the protein database with UniProt annotations."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="proteins.duckdb",
        help="Path to the DuckDB database (default: proteins.duckdb).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache/uniprot",
        help="Directory for raw response cache (default: cache/uniprot).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Accessions per request, max 100 (UniProt OR limit) (default: 100).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max accessions to process this run (for testing/resume).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per request (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  UniProt Annotation Enrichment")
    print("=" * 60)
    print(f"  Database      : {db_path}")
    print(f"  Cache dir     : {args.cache_dir}")
    print(f"  Batch size    : {args.batch_size:,}")
    if args.limit:
        print(f"  Limit         : {args.limit:,}")
    print("-" * 60)

    try:
        summary = annotate_database(
            db_path=db_path,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            max_retries=args.max_retries,
        )
        print(format_summary(summary))
    except Exception as exc:
        logger.exception("Enrichment failed")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()