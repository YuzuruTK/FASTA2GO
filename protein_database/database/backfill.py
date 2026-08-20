"""Backfill/recompute locally-computable features and composition.

The ``sequence_features`` and ``amino_acid_composition`` tables are derived
purely from each protein's FASTA sequence (physicochemical properties and
amino-acid percentages). These are not provided by the UniProt API, so any
gaps in those tables can only be repaired by recomputing them locally from
the stored ``proteins.sequence`` column.

By default this routine only processes proteins that are **missing** rows in
those child tables (an anti-join), so it stays fast even on large databases.
Set ``force=True`` to recompute/overwrite all proteins instead.
"""

from __future__ import annotations

import duckdb
from tqdm import tqdm

from protein_database.database.connection import bulk_upsert
from protein_database.database.models import (
    AminoAcidComposition as CompositionModel,
    SequenceFeatures as SequenceFeaturesModel,
)
from protein_database.features.composition import compute_amino_acid_composition
from protein_database.features.sequence_features import compute_sequence_features
from protein_database.pipeline import (
    _COMPOSITION_COLUMNS,
    _FEATURES_COLUMNS,
    _composition_row,
    _features_row,
)


def _missing_rows(conn: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str, str]]:
    """Return (fasta_id, source_file, sequence) for proteins missing a child row.

    Args:
        conn: DuckDB connection.
        table: Child table name (sequence_features or amino_acid_composition).

    Returns:
        List of (fasta_id, source_file, sequence) tuples missing in ``table``.
    """
    return [
        (r[0], r[1], r[2])
        for r in conn.execute(
            f"""
            SELECT p.fasta_id, p.source_file, p.sequence
            FROM proteins p
            LEFT JOIN {table} t
                ON p.fasta_id = t.fasta_id AND p.source_file = t.source_file
            WHERE t.fasta_id IS NULL
            """
        ).fetchall()
    ]


def _all_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """Return (fasta_id, source_file, sequence) for every protein."""
    return [
        (r[0], r[1], r[2])
        for r in conn.execute(
            "SELECT fasta_id, source_file, sequence FROM proteins"
        ).fetchall()
    ]


def backfill_features(
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 1000,
    force: bool = False,
) -> dict[str, int]:
    """Recompute and write sequence features and amino-acid composition.

    By default, only proteins missing rows in ``sequence_features`` and/or
    ``amino_acid_composition`` are recomputed (gap-filling). If ``force`` is
    True, every protein is recomputed and overwritten.

    Args:
        conn: DuckDB connection.
        batch_size: Number of proteins to upsert per batch.
        force: If True, recompute all proteins (overwrite). If False (default),
            only fill rows that are missing.

    Returns:
        Dict with counts: ``features_written``, ``compositions_written``,
        ``errors``, ``total``.
    """
    if force:
        feature_rows = _all_rows(conn)
        composition_rows = _all_rows(conn)
        total = len(feature_rows)
    else:
        feature_rows = _missing_rows(conn, "sequence_features")
        composition_rows = _missing_rows(conn, "amino_acid_composition")
        total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]

    # Unified workload: key -> (sequence, needs_features, needs_composition).
    need_features: dict[tuple[str, str], str] = {r[:2]: r[2] for r in feature_rows}
    need_composition: dict[tuple[str, str], str] = {r[:2]: r[2] for r in composition_rows}
    keys = set(need_features) | set(need_composition)

    def _build_feature(fasta_id: str, source_file: str, sequence: str) -> list:
        features = compute_sequence_features(sequence)
        model = SequenceFeaturesModel(
            fasta_id=fasta_id, source_file=source_file, **features.__dict__
        )
        return _features_row(model)

    def _build_composition(fasta_id: str, source_file: str, sequence: str) -> list:
        composition = compute_amino_acid_composition(sequence)
        model = CompositionModel(
            fasta_id=fasta_id, source_file=source_file, **composition.__dict__
        )
        return _composition_row(model)

    def _upsert(table: str, columns: list[str], rows: list[list]) -> None:
        if not rows:
            return
        bulk_upsert(
            conn,
            table,
            columns,
            rows,
            on_conflict_columns=["fasta_id", "source_file"],
            update_columns=columns[2:],
        )

    batch_features: list[list] = []
    batch_compositions: list[list] = []
    features_written = 0
    compositions_written = 0
    errors = 0

    def _flush() -> None:
        nonlocal features_written, compositions_written
        if batch_features:
            _upsert("sequence_features", _FEATURES_COLUMNS, batch_features)
            features_written += len(batch_features)
        if batch_compositions:
            _upsert("amino_acid_composition", _COMPOSITION_COLUMNS, batch_compositions)
            compositions_written += len(batch_compositions)
        batch_features.clear()
        batch_compositions.clear()

    progress = tqdm(
        total=len(keys),
        desc="Backfilling",
        unit="protein",
        leave=True,
    )

    try:
        for key in keys:
            fasta_id, source_file = key
            sequence = need_features[key] if key in need_features else need_composition[key]
            try:
                if key in need_features:
                    batch_features.append(_build_feature(fasta_id, source_file, sequence))
                if key in need_composition:
                    batch_compositions.append(_build_composition(fasta_id, source_file, sequence))
                if len(batch_features) >= batch_size or len(batch_compositions) >= batch_size:
                    _flush()
            except Exception:  # noqa: BLE001
                errors += 1
            progress.update(1)

        _flush()
    finally:
        progress.close()

    return {
        "features_written": features_written,
        "compositions_written": compositions_written,
        "errors": errors,
        "total": total,
    }
