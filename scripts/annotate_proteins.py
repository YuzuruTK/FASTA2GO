#!/usr/bin/env python3
"""CLI entry point for UniProt annotation enrichment.

Usage:
    uv run python scripts/annotate_proteins.py --db proteins.duckdb
    uv run python scripts/annotate_proteins.py --db proteins.duckdb --source cache
    uv run python scripts/annotate_proteins.py --db proteins.duckdb --source api --only-missing

This script enriches the annotations of proteins that are already in the
database. It does not download FASTA sequences.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.uniprot import MAX_ACCESSIONS_PER_REQUEST, annotate_database


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
        help="Directory for the snapshot, staging files and any legacy TSVs "
        "(default: cache/uniprot).",
    )
    parser.add_argument(
        "--source", type=str, choices=["bulk", "cache", "api"], default="bulk",
        help="bulk downloads the reviewed-proteome snapshot; cache replays local "
        "TSV files only; api fetches only the missing or stale accessions from "
        "the UniProt REST API. Default: bulk.",
    )
    parser.add_argument(
        "--refresh-snapshot", action="store_true",
        help="Re-download the bulk snapshot even when cached.",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="API mode: only proteins that have no annotation row yet.",
    )
    parser.add_argument(
        "--older-than-days", type=int, default=None,
        help="API mode: also refresh annotations older than N days. This is also "
        "how proteins recorded as not_found are retried.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="API mode: maximum number of local proteins to select.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=MAX_ACCESSIONS_PER_REQUEST,
        help="API mode: accessions per request, 1..1000 "
        f"(default: {MAX_ACCESSIONS_PER_REQUEST}).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="API mode: retries per request on 429/5xx and network errors "
        "(default: 5).",
    )
    parser.add_argument(
        "--base-delay", type=float, default=1.0,
        help="API mode: initial retry backoff in seconds (default: 1.0).",
    )
    args = parser.parse_args()

    if not 1 <= args.chunk_size <= MAX_ACCESSIONS_PER_REQUEST:
        parser.error(
            f"--chunk-size must be between 1 and {MAX_ACCESSIONS_PER_REQUEST} "
            "(the UniProt per-request limit)."
        )
    if args.source != "api" and (
        args.only_missing or args.older_than_days is not None or args.limit
    ):
        parser.error("--only-missing/--older-than-days/--limit require --source api.")

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
    if args.source == "api":
        filters = [
            name
            for name, enabled in (
                ("only-missing", args.only_missing),
                (
                    f"older-than-days={args.older_than_days}",
                    args.older_than_days is not None,
                ),
                (f"limit={args.limit}", bool(args.limit)),
            )
            if enabled
        ]
        selector = ", ".join(filters) if filters else "all"
        print(f"  Select   : {selector}")
        print(
            f"  Requests : chunk={args.chunk_size}, retries={args.max_retries}, "
            f"backoff={args.base_delay}s"
        )
    print("-" * 60)

    try:
        summary = annotate_database(
            db_path=db_path,
            cache_dir=args.cache_dir,
            source=args.source,
            refresh_snapshot=args.refresh_snapshot,
            only_missing=args.only_missing,
            older_than_days=args.older_than_days,
            limit=args.limit,
            chunk_size=args.chunk_size,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    mode = summary["mode"]
    failed = summary["failed"]
    total = summary["total_local_proteins"]
    handled = summary["accessions"]
    annotated = summary["annotated"]
    missing = summary["missing_function"]
    not_found = summary["not_found"]
    requests_made = summary["http_requests"]
    coverage = summary["function_coverage_pct"]
    print("=" * 60)
    print("  UniProt Annotation Summary")
    print("=" * 60)
    print(f"  Mode                  : {mode}")
    print(f"  Total local proteins  : {total:,}")
    print(f"  Accessions handled    : {handled:,}")
    print(f"  Annotated rows        : {annotated:,}")
    print(f"  Missing Function      : {missing:,}")
    print(f"  Not found             : {not_found:,}")
    print(f"  HTTP requests         : {requests_made:,}")
    print(f"  Function coverage     : {coverage:.2f}%")
    print("=" * 60)

    if failed:
        print(
            f"Error: {failed:,} accession(s) could not be fetched. "
            "Nothing was written for them and they are not marked as not_found; "
            "run the same command again to retry.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
