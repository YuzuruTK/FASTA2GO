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
from protein_database.uniprot.ingest import bulk_annotate_database

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
        f"  Elapsed time                : {s.elapsed_seconds:.1f}s",
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
        "--source",
        type=str,
        choices=["bulk", "cache", "api"],
        default="bulk",
        help="Annotation source: 'bulk' (single streaming snapshot download, "
        "merging the legacy cache when present), 'cache' (offline replay of "
        "the legacy per-batch *.tsv cache), or 'api' (batched REST retrieval "
        "with resume). Default: bulk.",
    )
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="Re-download the bulk snapshot even when one exists (bulk only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Accessions per request, max 100 (UniProt OR limit, api source only) (default: 100).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max accessions to process this run (for testing/resume, api source only).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per request (api source only, default: 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent HTTP-fetch threads (api source only, default: 4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--bulk-dir",
        type=str,
        default=None,
        help="Directory for bulk snapshot chunks (default: <cache-dir>/bulk).",
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
    print(f"  Source        : {args.source}")
    print(f"  Cache dir     : {args.cache_dir}")
    if args.source == "api":
        print(f"  Batch size    : {args.batch_size:,}")
        print(f"  Workers       : {args.workers}")
        if args.limit:
            print(f"  Limit         : {args.limit:,}")
    if args.source == "bulk":
        print(f"  Bulk dir      : {args.bulk_dir or str(Path(args.cache_dir) / 'bulk')}")
        print(f"  Refresh       : {'Yes' if args.refresh_snapshot else 'No'}")
    print("-" * 60)

    try:
        if args.source == "api":
            summary = annotate_database(
                db_path=db_path,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                limit=args.limit,
                max_retries=args.max_retries,
                workers=args.workers,
            )
        else:
            summary = bulk_annotate_database(
                db_path=db_path,
                cache_dir=args.cache_dir,
                source=args.source,
                refresh_snapshot=args.refresh_snapshot,
                bulk_dir=args.bulk_dir,
            )
        print(format_summary(summary))
    except Exception as exc:
        logger.exception("Enrichment failed")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()