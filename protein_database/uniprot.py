"""UniProt enrichment: one streaming TSV download, ingested natively.

A single request to the UniProtKB ``stream`` endpoint delivers the whole
reviewed proteome as gzipped TSV. DuckDB's native ``read_csv`` ingests the
file(s) into a temp staging table, and one set-based ``INSERT ... ON
CONFLICT`` fans the annotations out to every matching local protein row.
"""

import glob as glob_module
from pathlib import Path

import duckdb
import requests

from protein_database import db, pipeline
from protein_database.schema import ANNOTATION_COLUMNS

STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
REVIEWED_QUERY = "reviewed:true"
SNAPSHOT_NAME = "uniprot_sprot.tsv.gz"

#: Verified UniProtKB return-field names (must match the header, in order).
UNIPROT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "cc_function",
    "cc_subcellular_location",
    "protein_existence",
    "organism_name",
    "keyword",
]

#: UniProt TSV header -> database annotation column.
TSV_MAP = {
    "Entry": "uniprot_accession",
    "Entry Name": "uniprot_id",
    "Protein names": "protein_name",
    "Gene Names": "gene_name",
    "Function [CC]": "function",
    "Subcellular location [CC]": "subcellular_location",
    "Protein existence": "protein_existence",
    "Organism": "organism",
    "Keywords": "keywords",
}

TSV_TYPES = ", ".join(f"'{header}': 'VARCHAR'" for header in TSV_MAP)
STAGING_TABLE = "uniprot_staging"


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def download_snapshot(
    bulk_dir: str | Path = "cache/uniprot/bulk", refresh: bool = False
) -> Path:
    """Download the reviewed-proteome TSV once and reuse it thereafter.

    The body is streamed to a ``.part`` file and renamed only after the
    transfer completes, so an interrupted run is never mistaken for a
    complete snapshot.
    """
    dest = Path(bulk_dir) / SNAPSHOT_NAME
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        STREAM_URL,
        params={
            "query": REVIEWED_QUERY,
            "format": "tsv",
            "fields": ",".join(UNIPROT_FIELDS),
            "compressed": "true",
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        with open(part, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    part.rename(dest)
    return dest


def _build_staging_table(conn: duckdb.DuckDBPyConnection, files: list[str]) -> int:
    """Ingest TSV file(s) into a deduped temp table (earlier files win)."""
    sources = "[" + ", ".join(_quote(path) for path in files) + "]"
    selected = ", ".join(f'"{header}"' for header in TSV_MAP)
    aggregated = ", ".join(
        f'arg_min("{header}", _src) AS "{header}"'
        for header in TSV_MAP
        if header != "Entry"
    )
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE {STAGING_TABLE} AS "
        f'SELECT "Entry", {aggregated} FROM ('
        f"SELECT filename AS _src, {selected} FROM read_csv({sources}, "
        f"delim='\\t', header=true, union_by_name=true, filename=true, "
        f"columns={{{TSV_TYPES}}})"
        f') WHERE "Entry" IS NOT NULL GROUP BY "Entry"'
    )
    return conn.execute(f"SELECT count(*) FROM {STAGING_TABLE}").fetchone()[0]


def _upsert_annotations(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Fan staged rows out to local proteins; return (written, missing_fn)."""
    columns = ", ".join(ANNOTATION_COLUMNS)
    updates = ", ".join(
        f"{column} = excluded.{column}"
        for column in ANNOTATION_COLUMNS
        if column not in ("fasta_id", "source_file")
    )
    select = (
        "p.fasta_id, p.source_file, "
        's."Entry", s."Entry Name", s."Protein names", s."Gene Names", '
        's."Function [CC]", s."Subcellular location [CC]", '
        's."Protein existence", s."Organism", s."Keywords", '
        "CASE WHEN s.\"Function [CC]\" IS NULL "
        "OR trim(s.\"Function [CC]\") = '' "
        "THEN 'missing_function' ELSE 'fetched' END, now()"
    )
    conn.execute(
        f"INSERT INTO uniprot_annotations ({columns}) SELECT {select} "
        f"FROM proteins p JOIN {STAGING_TABLE} s "
        f"ON split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        f"AND upper(split_part(p.fasta_id, '|', 2)) = s.\"Entry\" "
        f"ON CONFLICT (fasta_id, source_file) DO UPDATE SET {updates}"
    )
    written = conn.execute(
        "SELECT count(*) FROM uniprot_annotations "
        "WHERE status IN ('fetched', 'missing_function')"
    ).fetchone()[0]
    missing = conn.execute(
        "SELECT count(*) FROM uniprot_annotations WHERE status = 'missing_function'"
    ).fetchone()[0]
    return written, missing


def annotate_database(
    db_path: str | Path = "proteins.duckdb",
    cache_dir: str | Path = "cache/uniprot",
    source: str = "bulk",
    refresh_snapshot: bool = False,
) -> dict:
    """Enrich the protein database with UniProt annotations.

    Args:
        db_path: Path to the DuckDB database.
        cache_dir: Directory for the snapshot and any legacy ``*.tsv`` files.
        source: ``'bulk'`` downloads the snapshot (plus any legacy cache),
            ``'cache'`` ingests only local ``*.tsv``/``*.tsv.gz`` files.
        refresh_snapshot: Re-download the bulk snapshot even when cached.

    Returns:
        Dict with run statistics.
    """
    cache = Path(cache_dir)
    if source == "bulk":
        files = [str(download_snapshot(cache / "bulk", refresh_snapshot))]
        # Merge any legacy per-batch TSVs alongside the snapshot.
        files += sorted(glob_module.glob(str(cache / "*.tsv")))
    elif source == "cache":
        # Offline replay of every TSV/TSV.gz under the cache (batches + chunks).
        files = sorted(glob_module.glob(str(cache / "**" / "*.tsv*"), recursive=True))
    else:
        raise ValueError(f"Unknown source: {source!r} (expected 'bulk' or 'cache').")
    if not files:
        raise ValueError("No UniProt TSV sources found.")

    conn = db.connect(db_path)
    try:
        db.init_database(conn)

        print("Backfilling sequence features and amino-acid composition...")
        report = pipeline.backfill_features(conn)
        print(f"Backfilled {report['written']:,} proteins ({report['errors']} errors).")

        staged = _build_staging_table(conn, files)
        print(f"Staged {staged:,} distinct accessions from {len(files)} file(s).")

        written, missing_function = _upsert_annotations(conn)
        print(f"Upserted {written:,} annotation rows ({missing_function:,} missing function).")

        total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
        with_function = conn.execute(
            "SELECT count(*) FROM uniprot_annotations "
            "WHERE function IS NOT NULL AND function <> ''"
        ).fetchone()[0]
        not_found = conn.execute(
            f"SELECT count(DISTINCT upper(split_part(fasta_id, '|', 2))) "
            f"FROM proteins WHERE split_part(fasta_id, '|', 1) IN ('sp', 'tr') "
            f"AND upper(split_part(fasta_id, '|', 2)) NOT IN "
            f'(SELECT "Entry" FROM {STAGING_TABLE})'
        ).fetchone()[0]
        return {
            "total_local_proteins": total,
            "annotated": written,
            "missing_function": missing_function,
            "not_found": not_found,
            "function_coverage_pct": (with_function / total * 100.0) if total else 0.0,
            "files": len(files),
        }
    finally:
        conn.close()
