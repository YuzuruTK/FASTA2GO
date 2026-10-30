#!/usr/bin/env python3
"""FASTA2GO - etapa 1: extração (FASTA -> features -> UniProt -> DuckDB).

    uv run python etapa1/main.py

Os ajustes vêm de ``etapa1/fasta2go.config``; não há argumentos de linha de
comando. A extração começa de um banco vazio: um arquivo existente não é
substituído nem apagado, e a mensagem de erro pede que ele seja removido.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fasta2go import db, pipeline, uniprot  # noqa: E402

CONFIG_FILE = Path(__file__).resolve().parent / "fasta2go.config"
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_FILE, encoding="utf-8")

FASTA = PROJECT_ROOT / CONFIG.get(
    "extraction", "fasta", fallback="data/uniprot_sprot.fasta"
)
DATABASE = PROJECT_ROOT / CONFIG.get(
    "extraction", "database", fallback="db/fasta2go.duckdb"
)
BATCH_SIZE = CONFIG.getint("extraction", "batch_size", fallback=5000)
ACCESSIONS_PER_REQUEST = CONFIG.getint(
    "extraction", "accessions_per_request", fallback=200
)
LIMIT = CONFIG.getint("extraction", "limit", fallback=None)  # sem chave = todas
WITH_UNIPROT = CONFIG.getboolean("extraction", "uniprot", fallback=True)


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



def main() -> None:
    """Extract the FASTA into a new database, batch by batch."""
    path = Path(DATABASE)
    if path.exists():
        sys.exit(f"Database already exists: {path} (delete it to extract from zero)")

    conn = db.connect(path)
    try:
        db.init_schema(conn)
        print(f"Database: {path.resolve()}")
        print(f"\nExtracting {FASTA} in batches of {BATCH_SIZE:,}")
        counts = pipeline.extract(
            conn,
            FASTA,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            accessions_per_request=ACCESSIONS_PER_REQUEST,
            with_uniprot=WITH_UNIPROT,
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
        if not WITH_UNIPROT:
            print("  UniProt enrichment skipped (uniprot = false)")
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


if __name__ == "__main__":
    try:
        main()
    except (uniprot.UniProtError, FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")
