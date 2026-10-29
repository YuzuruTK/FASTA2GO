#!/usr/bin/env python3
"""FASTA2GO - extraction stage.

    uv run python main.py run       # ingest -> features -> UniProt -> report
    uv run python main.py status    # what the database contains

Both commands are safe to repeat: work already done is not repeated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fasta2go import db, pipeline, uniprot

DEFAULT_FASTA = "data/uniprot_sprot.fasta"
DEFAULT_DB = "db/fasta2go.duckdb"


def format_summary(summary: dict) -> list[str]:
    """Human-readable lines for the database summary."""
    lines = [
        f"  proteins                  {summary['proteins']:,}",
        f"    without UniProt acc.    {summary['without_accession']:,}",
        f"  with features             {summary['with_features']:,}"
        f"  (versions: {summary['feature_versions'] or '-'})",
        f"  with composition          {summary['with_composition']:,}",
        f"  with dipeptides           {summary['with_dipeptides']:,}",
        f"  UniProt matched           {summary['matched']:,}",
        f"  UniProt not found         {summary['not_found']:,}",
        f"  not queried yet           {summary['unqueried']:,}",
        f"  GO annotations            {summary['go_rows']:,} rows"
        f" on {summary['proteins_with_go']:,} proteins",
    ]
    aspects = ", ".join(
        f"{aspect}={count:,}" for aspect, count in summary["go_by_aspect"].items()
    )
    lines.append(f"    by aspect               {aspects or '-'}")
    issues = ", ".join(
        f"{issue}={count:,}" for issue, count in summary["issues"].items()
    )
    lines.append(f"  ingestion issues          {issues or '-'}")
    lines.append(f"  orphan rows               {summary['orphans']:,}")
    return lines


def run(args: argparse.Namespace) -> None:
    """Ingest, compute features and enrich, printing each step's result."""
    conn = db.connect(args.db)
    try:
        db.init_schema(conn)
        print(f"Database: {Path(args.db).resolve()}")

        print(f"\n[1/3] Ingesting {args.fasta}")
        counts = pipeline.ingest(conn, args.fasta, limit=args.limit)
        print(
            f"  {counts['inserted']:,} inserted, "
            f"{counts['already_present']:,} already present, "
            f"{counts['rejected']:,} rejected, "
            f"{counts['duplicates']:,} duplicates "
            f"({counts['records']:,} records, {counts['elapsed']}s)"
        )

        print("\n[2/3] Computing sequence features")
        counts = pipeline.compute_features(conn)
        print(
            f"  {counts['computed']:,} computed, {counts['skipped']:,} without "
            f"canonical residues ({counts['elapsed']}s)"
        )

        if args.no_uniprot:
            print("\n[3/3] UniProt enrichment skipped (--no-uniprot)")
        else:
            print("\n[3/3] Enriching with UniProt")
            counts = pipeline.enrich(
                conn,
                batch_size=args.batch_size,
                max_age_days=args.max_age_days,
                limit=args.limit,
            )
            print(
                f"  {counts['queried']:,} proteins queried in "
                f"{counts['requests']} request(s): {counts['matched']:,} matched, "
                f"{counts['not_found']:,} not found, "
                f"{counts['go_rows']:,} GO rows ({counts['elapsed']}s)"
            )

        print("\nDatabase summary")
        print("\n".join(format_summary(pipeline.database_summary(conn))))
    finally:
        conn.close()


def status(args: argparse.Namespace) -> None:
    """Print what the database contains, without processing anything."""
    if not Path(args.db).exists():
        sys.exit(f"Database not found: {args.db}")
    conn = db.connect(args.db)
    try:
        db.init_schema(conn)
        print(f"Database: {Path(args.db).resolve()}")
        print("\n".join(format_summary(pipeline.database_summary(conn))))
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FASTA2GO extraction: FASTA -> DuckDB with features, "
        "UniProt metadata and GO annotations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="ingest, compute features, enrich")
    run_parser.add_argument("--fasta", default=DEFAULT_FASTA, help="input FASTA (.gz ok)")
    run_parser.add_argument("--db", default=DEFAULT_DB, help="DuckDB file")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N proteins (for a quick run)",
    )
    run_parser.add_argument(
        "--no-uniprot", action="store_true", help="skip the UniProt step"
    )
    run_parser.add_argument(
        "--max-age-days",
        type=int,
        default=None,
        help="re-query annotations older than N days (default: never)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="accessions per UniProt request",
    )
    run_parser.set_defaults(handler=run)

    status_parser = commands.add_parser("status", help="report database contents")
    status_parser.add_argument("--db", default=DEFAULT_DB, help="DuckDB file")
    status_parser.set_defaults(handler=status)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (uniprot.UniProtError, FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
