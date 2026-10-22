"""Pipeline: FASTA parse -> feature extraction -> DuckDB storage."""

import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from protein_database import db
from protein_database.fasta import parse_fasta
from protein_database.progress import track
from protein_database.features import composition, sequence_features
from protein_database.schema import (
    COMPOSITION_COLUMNS,
    FEATURE_COLUMNS,
    INPUT_COLUMNS,
    PROTEIN_COLUMNS,
)

FEATURE_TABLE_COLUMNS = INPUT_COLUMNS + FEATURE_COLUMNS
COMPOSITION_TABLE_COLUMNS = INPUT_COLUMNS + COMPOSITION_COLUMNS


def _feature_rows(sequence: str) -> tuple[list[float], list[float]]:
    """Return (feature values, composition values) in schema column order."""
    features = sequence_features(sequence)
    amino = composition(sequence)
    return (
        [features[column] for column in FEATURE_COLUMNS],
        [amino[column] for column in COMPOSITION_COLUMNS],
    )


def _insert_features(
    conn: duckdb.DuckDBPyConnection,
    feature_rows: list[list],
    composition_rows: list[list],
    cache_dir: str | Path,
) -> None:
    db.bulk_insert(
        conn, "sequence_features", FEATURE_TABLE_COLUMNS, feature_rows, cache_dir
    )
    db.bulk_insert(
        conn,
        "amino_acid_composition",
        COMPOSITION_TABLE_COLUMNS,
        composition_rows,
        cache_dir,
    )


def process_fasta_file(
    fasta_path: str | Path,
    db_path: str | Path = "proteins.duckdb",
    batch_size: int = 1000,
    limit: int | None = None,
    overwrite: bool = False,
    csv_cache_dir: str | Path = "cache/csv_bulk",
) -> dict:
    """Parse a FASTA file, extract features, and store them in DuckDB.

    Rows already present for the source file are skipped, so re-runs are
    idempotent.

    Returns:
        Dict with keys 'processed', 'skipped', 'errors', 'elapsed_seconds'
        and 'database_path'.
    """
    start = time.time()
    source_file = Path(fasta_path).name

    conn = db.connect(db_path)
    db.init_database(conn, overwrite=overwrite)

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT fasta_id FROM proteins WHERE source_file = ?", [source_file]
        ).fetchall()
    }
    if existing:
        print(f"Skipping {len(existing):,} records already stored for '{source_file}'.")

    created_at = datetime.now(timezone.utc).isoformat()
    proteins: list[list] = []
    feature_rows: list[list] = []
    composition_rows: list[list] = []
    processed = skipped = errors = 0

    def flush() -> None:
        if not proteins:
            return
        db.bulk_insert(conn, "proteins", PROTEIN_COLUMNS, proteins, csv_cache_dir)
        _insert_features(conn, feature_rows, composition_rows, csv_cache_dir)
        proteins.clear()
        feature_rows.clear()
        composition_rows.clear()

    try:
        for record in track(parse_fasta(fasta_path), desc="Processing", unit="protein"):
            if limit is not None and processed + skipped >= limit:
                break
            if record.fasta_id in existing:
                skipped += 1
                continue
            try:
                features, amino = _feature_rows(record.sequence)
            except ValueError:
                errors += 1
                continue

            proteins.append(
                [
                    record.fasta_id,
                    source_file,
                    record.description,
                    record.sequence,
                    len(record.sequence),
                    created_at,
                ]
            )
            feature_rows.append([record.fasta_id, source_file, *features])
            composition_rows.append([record.fasta_id, source_file, *amino])
            processed += 1

            if len(proteins) >= batch_size:
                flush()
        flush()
    finally:
        conn.close()

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": time.time() - start,
        "database_path": str(db_path),
    }


def backfill_features(
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 1000,
    csv_cache_dir: str | Path = "cache/csv_bulk",
) -> dict[str, int]:
    """Recompute features for proteins missing a ``sequence_features`` row."""
    rows = conn.execute(
        """
        SELECT p.fasta_id, p.source_file, p.sequence
        FROM proteins p
        WHERE NOT EXISTS (
            SELECT 1 FROM sequence_features f
            WHERE f.fasta_id = p.fasta_id AND f.source_file = p.source_file
        )
        """
    ).fetchall()

    features: list[list] = []
    compositions: list[list] = []
    written = errors = 0

    def flush() -> None:
        nonlocal written
        if features:
            _insert_features(conn, features, compositions, csv_cache_dir)
            written += len(features)
            features.clear()
            compositions.clear()

    for fasta_id, source_file, sequence in track(
        rows, desc="Backfilling", unit="protein"
    ):
        try:
            feature_values, amino = _feature_rows(sequence)
        except ValueError:
            errors += 1
            continue
        features.append([fasta_id, source_file, *feature_values])
        compositions.append([fasta_id, source_file, *amino])
        if len(features) >= batch_size:
            flush()
    flush()

    return {"written": written, "errors": errors, "total": len(rows)}
