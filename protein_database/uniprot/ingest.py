"""Native set-based ingestion of UniProt TSV snapshots into DuckDB.

The bulk snapshot chunks and the legacy per-batch ``*.tsv`` cache already
hold the exact columns the database needs, so instead of parsing rows in
Python and writing them back line-by-line, DuckDB ``read_csv`` ingests the
files natively into a temp staging table (deduplicated per accession) and a
single ``INSERT ... SELECT ... ON CONFLICT`` fans the annotations out to
every matching local protein row.
"""

from __future__ import annotations

import glob as glob_module
from pathlib import Path

import duckdb

from protein_database.uniprot.annotate import (
    ANNOTATION_COLUMNS,
    ANNOTATION_UPDATE_COLUMNS,
    RunSummary,
    STATUS_MISSING_FUNCTION,
    compute_function_coverage,
)
from protein_database.uniprot.bulk import download_snapshot

#: DuckDB types for the nine UniProt TSV columns (VARCHAR each).
TSV_COLUMN_TYPES = {
    "Entry": "VARCHAR",
    "Entry Name": "VARCHAR",
    "Protein names": "VARCHAR",
    "Gene Names": "VARCHAR",
    "Function [CC]": "VARCHAR",
    "Subcellular location [CC]": "VARCHAR",
    "Protein existence": "VARCHAR",
    "Organism": "VARCHAR",
    "Keywords": "VARCHAR",
}

#: Name of the temp staging table holding native-imported TSV rows.
STAGING_TABLE = "uniprot_staging"


def _sql_quote(value: str | Path) -> str:
    """Quote a path/string literal for SQL embedding (single-quote escape)."""
    return "'" + str(value).replace("'", "''") + "'"


def expand_sources(sources: list[str | Path]) -> list[Path]:
    """Expand glob patterns/paths into existing files, preserving order.

    Args:
        sources: Ordered file paths or glob patterns (e.g. a bulk snapshot
            chunk list followed by a legacy ``*.tsv`` cache glob). Earlier
            sources win on duplicate accessions.

    Returns:
        Deduplicated list of existing file paths, in source order.
    """
    files: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        for match in sorted(glob_module.glob(str(source))):
            path = Path(match)
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def build_staging_table(
    conn: duckdb.DuckDBPyConnection,
    sources: list[str | Path],
    staging_table: str = STAGING_TABLE,
) -> int:
    """Natively ingest TSV snapshot files into a deduped temp staging table.

    Args:
        conn: DuckDB connection.
        sources: Ordered file paths/glob patterns (see :func:`expand_sources`).
        staging_table: Name of the temp table to create.

    Returns:
        Number of distinct accessions staged.

    Raises:
        ValueError: If no source file matches.
    """
    files = expand_sources(sources)
    if not files:
        raise ValueError("No TSV sources matched; nothing to stage.")
    cols_sql = ", ".join(f'"{name}"' for name in TSV_COLUMN_TYPES)
    types_sql = ", ".join(
        f"'{name}': '{dtype}'" for name, dtype in TSV_COLUMN_TYPES.items()
    )
    sources_sql = ", ".join(_sql_quote(p) for p in files)
    # DuckDB reads listed files in order, so earlier sources (bulk snapshot)
    # take precedence over later ones (legacy cache) on duplicate accessions.
    # The rank is folded into each column with arg_min: per accession, every
    # value comes from the row with the smallest source rank (deterministic,
    # unlike DISTINCT ON with a non-matching ORDER BY).
    value_cols = ", ".join(
        f'arg_min("{name}", _src) AS "{name}"'
        for name in TSV_COLUMN_TYPES
        if name != "Entry"
    )
    ranked: list[str] = []
    for rank, path in enumerate(files):
        ranked.append(
            f"SELECT {rank} AS _src, * FROM read_csv({_sql_quote(path)}, "
            f"delim='\\t', header=true, union_by_name=true, columns={{{types_sql}}})"
        )
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE {staging_table} AS "
        f'SELECT "Entry", {value_cols} '
        f"FROM ({' UNION ALL '.join(ranked)}) "
        f'GROUP BY "Entry"'
    )
    return conn.execute(f"SELECT count(*) FROM {staging_table}").fetchone()[0]


def count_local_matches(conn: duckdb.DuckDBPyConnection, staging_table: str = STAGING_TABLE) -> int:
    """Count local protein rows whose accession appears in the staging table."""
    return conn.execute(
        f"SELECT count(*) FROM proteins p "
        f"JOIN {staging_table} s "
        f"ON upper(split_part(p.fasta_id, '|', 2)) = s.\"Entry\" "
        f"WHERE split_part(p.fasta_id, '|', 1) IN ('sp', 'tr')"
    ).fetchone()[0]


def count_missing_accessions(conn: duckdb.DuckDBPyConnection, staging_table: str = STAGING_TABLE) -> int:
    """Count distinct local accessions absent from the staging table."""
    return conn.execute(
        f"SELECT count(DISTINCT upper(split_part(p.fasta_id, '|', 2))) "
        f"FROM proteins p LEFT JOIN {staging_table} s "
        f"ON upper(split_part(p.fasta_id, '|', 2)) = s.\"Entry\" "
        f"WHERE split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') AND s.\"Entry\" IS NULL"
    ).fetchone()[0]


def count_unique_local_accessions(conn: duckdb.DuckDBPyConnection) -> int:
    """Count distinct UniProt accessions referenced by local protein rows."""
    return conn.execute(
        "SELECT count(DISTINCT upper(split_part(fasta_id, '|', 2))) "
        "FROM proteins WHERE split_part(fasta_id, '|', 1) IN ('sp', 'tr')"
    ).fetchone()[0]


def upsert_staged_annotations(
    conn: duckdb.DuckDBPyConnection,
    staging_table: str = STAGING_TABLE,
) -> tuple[int, int]:
    """Fan staged TSV rows out to local proteins in one set-based upsert.

    Args:
        conn: DuckDB connection.
        staging_table: Temp table built by :func:`build_staging_table`.

    Returns:
        Tuple ``(rows_written, rows_missing_function)``.
    """
    col_list = ", ".join(ANNOTATION_COLUMNS)
    update_str = ", ".join(f"{c} = excluded.{c}" for c in ANNOTATION_UPDATE_COLUMNS)
    conn.execute(
        f"INSERT INTO uniprot_annotations ({col_list}) "
        f"SELECT p.fasta_id, p.source_file, s.\"Entry\", s.\"Entry Name\", "
        f"s.\"Protein names\", s.\"Gene Names\", s.\"Function [CC]\", "
        f"s.\"Subcellular location [CC]\", s.\"Protein existence\", "
        f"s.\"Organism\", s.\"Keywords\", "
        f"CASE WHEN s.\"Function [CC]\" IS NULL "
        f"OR trim(s.\"Function [CC]\") = '' "
        f"THEN '{STATUS_MISSING_FUNCTION}' ELSE 'fetched' END, "
        f"now() "
        f"FROM proteins p JOIN {staging_table} s "
        f"ON upper(split_part(p.fasta_id, '|', 2)) = s.\"Entry\" "
        f"WHERE split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        f"ON CONFLICT (fasta_id, source_file) DO UPDATE SET {update_str}"
    )
    written = conn.execute(
        "SELECT count(*) FROM uniprot_annotations "
        f"WHERE status IN ('fetched', '{STATUS_MISSING_FUNCTION}')"
    ).fetchone()[0]
    missing_fn = conn.execute(
        "SELECT count(*) FROM uniprot_annotations "
        f"WHERE status = '{STATUS_MISSING_FUNCTION}'"
    ).fetchone()[0]
    return written, missing_fn


def bulk_annotate_database(
    db_path: str | Path = "proteins.duckdb",
    cache_dir: str | Path = "cache/uniprot",
    source: str = "bulk",
    refresh_snapshot: bool = False,
    bulk_dir: str | Path | None = None,
) -> RunSummary:
    """Enrich the protein database from TSV snapshots without HTTP batches.

    The bulk snapshot (single streaming-request download) and/or the legacy
    per-batch ``*.tsv`` cache are ingested natively by DuckDB, then a single
    set-based ``INSERT ... SELECT ... ON CONFLICT`` writes the annotations.
    No rows are parsed in Python and nothing is written line-by-line.

    Args:
        db_path: Path to the DuckDB database.
        cache_dir: Directory holding the legacy per-batch ``*.tsv`` cache.
        source: ``'bulk'`` (snapshot download, merging the legacy cache when
            present), ``'cache'`` (offline replay of the legacy cache only).
        refresh_snapshot: Re-download the snapshot even when one exists.
        bulk_dir: Directory for snapshot chunks (defaults to
            ``<cache_dir>/bulk``).

    Returns:
        RunSummary with run statistics.

    Raises:
        ValueError: For an unknown ``source`` or when no TSV files exist.
    """
    import time

    from protein_database.database.connection import init_database
    from protein_database.database.backfill import backfill_features

    start = time.time()
    summary = RunSummary()
    conn = duckdb.connect(str(db_path)) if db_path != ":memory:" else duckdb.connect(":memory:")
    try:
        init_database(conn)

        print("Backfilling sequence features and amino-acid composition...")
        backfill_report = backfill_features(conn)
        print(
            f"Backfilled features for {backfill_report['features_written']:,} "
            f"proteins and composition for {backfill_report['compositions_written']:,} "
            f"proteins ({backfill_report['errors']} errors)."
        )

        summary.total_local_proteins = conn.execute(
            "SELECT count(*) FROM proteins"
        ).fetchone()[0]
        summary.unique_accessions = count_unique_local_accessions(conn)
        print(
            f"Found {summary.total_local_proteins:,} local protein rows, "
            f"{summary.unique_accessions:,} unique accessions."
        )

        bulk_path = Path(bulk_dir) if bulk_dir else Path(cache_dir) / "bulk"
        legacy_glob = str(Path(cache_dir) / "*.tsv")
        sources: list[str | Path]
        if source == "bulk":
            print("Ensuring bulk snapshot (single streaming download if missing)...")
            chunks = download_snapshot(dest_dir=bulk_path, refresh=refresh_snapshot)
            summary.http_requests = 0 if not refresh_snapshot and len(chunks) else len(chunks)
            sources = [*chunks, legacy_glob]
        elif source == "cache":
            summary.http_requests = 0
            sources = [legacy_glob]
        else:
            raise ValueError(f"Unknown source: {source!r} (expected 'bulk' or 'cache').")

        staged = build_staging_table(conn, sources)
        print(f"Staged {staged:,} distinct accessions natively.")
        summary.cached_batches = len(expand_sources([legacy_glob]))

        summary.successfully_retrieved, summary.missing_function = upsert_staged_annotations(conn)
        summary.not_found = count_missing_accessions(conn)
        print(
            f"Upserted annotations for {summary.successfully_retrieved:,} local rows "
            f"({summary.missing_function:,} missing function)."
        )

        summary.function_coverage_pct = compute_function_coverage(conn)
        summary.elapsed_seconds = time.time() - start
        return summary
    finally:
        conn.close()
