#!/usr/bin/env python3
"""FASTA2GO - extraction stage.

    uv run python main.py run       # FASTA -> features -> UniProt -> DuckDB
    uv run python main.py status    # what the database contains

The extraction reads the FASTA in batches and stores each batch completely
before reading the next one. It always starts from an empty database: an
existing file is only replaced when ``--recreate`` is given.
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
    """Extract the FASTA into a new database, batch by batch."""
    path = Path(args.db)
    if path.exists():
        if not args.recreate:
            sys.exit(
                f"Database already exists: {path} "
                "(pass --recreate to extract from zero)"
            )
        # A full extraction starts from an empty database, so the previous file
        # (and a WAL left by an interrupted DuckDB) is replaced on purpose here.
        for stale in (path, path.with_name(path.name + ".wal")):
            stale.unlink(missing_ok=True)

    conn = db.connect(args.db)
    try:
        db.init_schema(conn)
        print(f"Database: {path.resolve()}")

        print(f"\nExtracting {args.fasta} in batches of {args.feature_batch_size:,}")
        counts = pipeline.extract(
            conn,
            args.fasta,
            batch_size=args.feature_batch_size,
            limit=args.limit,
            accessions_per_request=args.batch_size,
            with_uniprot=not args.no_uniprot,
        )
        print(
            f"  {counts['inserted']:,} proteins stored in {counts['batches']:,} "
            f"batch(es) ({counts['records']:,} records read, "
            f"{counts['rejected']:,} rejected, {counts['duplicates']:,} duplicates, "
            f"{counts['elapsed']}s)"
        )
        print(
            f"  {counts['computed']:,} feature sets computed, {counts['skipped']:,} "
            "without canonical residues"
        )
        if args.no_uniprot:
            print("  UniProt enrichment skipped (--no-uniprot)")
        else:
            print(
                f"  {counts['queried']:,} proteins queried in "
                f"{counts['requests']:,} request(s): {counts['matched']:,} matched, "
                f"{counts['not_found']:,} not found, {counts['go_rows']:,} GO rows"
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

    run_parser = commands.add_parser(
        "run", help="extract the FASTA into a new database, batch by batch"
    )
    run_parser.add_argument("--fasta", default=DEFAULT_FASTA, help="input FASTA (.gz ok)")
    run_parser.add_argument("--db", default=DEFAULT_DB, help="DuckDB file")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="store at most N proteins (for a quick run)",
    )
    run_parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=pipeline.DEFAULT_BATCH_SIZE,
        help="proteins per extraction batch (time/memory trade-off)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=pipeline.DEFAULT_ACCESSIONS_PER_REQUEST,
        help="accessions per UniProt request",
    )
    run_parser.add_argument(
        "--no-uniprot", action="store_true", help="store proteins and features only"
    )
    run_parser.add_argument(
        "--recreate",
        action="store_true",
        help="replace an existing database (a full extraction starts from zero)",
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
