"""Extraction pipeline: ingest -> features -> UniProt -> summary.

Every step is incremental and idempotent: it selects only what is missing or
outdated, writes in batches, and can be re-run after any interruption without
re-importing or recomputing what is already valid.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import duckdb
import requests

from fasta2go import db, features, uniprot
from fasta2go.fasta import read_fasta

KEY = ("source_file", "fasta_id")

PROTEIN_COLUMNS = [
    "source_file",
    "fasta_id",
    "fasta_header",
    "uniprot_accession",
    "sequence",
    "sequence_length",
    "ingested_at",
]
ISSUE_COLUMNS = [
    "source_file",
    "line_number",
    "fasta_id",
    "issue",
    "detail",
    "detected_at",
]
FEATURE_TABLE_COLUMNS = [
    "source_file",
    "fasta_id",
    *features.FEATURE_COLUMNS,
    "computed_at",
]
AAC_TABLE_COLUMNS = ["source_file", "fasta_id", *features.AAC_COLUMNS]
DPC_TABLE_COLUMNS = ["source_file", "fasta_id", *features.DPC_COLUMNS]
ANNOTATION_COLUMNS = [
    "source_file",
    "fasta_id",
    "uniprot_accession",
    "uniprot_id",
    "protein_name",
    "gene_names",
    "organism",
    "taxon_id",
    "protein_existence",
    "secondary_accessions",
    "keywords",
    "function_description",
    "subcellular_location",
    "source_length",
    "status",
    "retrieved_at",
]
GO_COLUMNS = [
    "source_file",
    "fasta_id",
    "go_id",
    "aspect",
    "go_name",
    "evidence_code",
    "evidence_source",
    "provenance",
    "retrieved_at",
]


def utc_now() -> datetime:
    """Naive UTC timestamp (DuckDB ``TIMESTAMP`` carries no time zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ingest(
    conn: duckdb.DuckDBPyConnection,
    fasta_path: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 2000,
) -> dict[str, Any]:
    """Import every valid record of a FASTA file.

    Proteins already stored for the same file are skipped, so the import is
    idempotent. Records that cannot be stored are counted and written to
    ``ingestion_issues`` instead of disappearing silently.
    """
    started = time.time()
    path = Path(fasta_path)
    source_file = path.name
    now = utc_now()
    known = {
        row[0]
        for row in conn.execute(
            "SELECT fasta_id FROM proteins WHERE source_file = ?", [source_file]
        ).fetchall()
    }
    conn.execute("DELETE FROM ingestion_issues WHERE source_file = ?", [source_file])

    counts = {
        "records": 0,
        "inserted": 0,
        "already_present": 0,
        "rejected": 0,
        "duplicates": 0,
    }
    rows: list[list[Any]] = []
    issues: list[list[Any]] = []
    seen: set[str] = set()

    def flush() -> None:
        nonlocal rows, issues
        counts["inserted"] += db.write_rows(
            conn, "proteins", PROTEIN_COLUMNS, rows, key=KEY
        )
        db.write_rows(conn, "ingestion_issues", ISSUE_COLUMNS, issues)
        rows, issues = [], []

    for record in read_fasta(path):
        counts["records"] += 1

        if record.fasta_id in seen:
            counts["duplicates"] += 1
            issues.append(
                [
                    source_file,
                    record.line_number,
                    record.fasta_id,
                    "duplicate_id",
                    "kept the first occurrence",
                    now,
                ]
            )
            continue
        seen.add(record.fasta_id)
        issues.extend(
            [source_file, record.line_number, record.fasta_id, code, detail, now]
            for code, detail in record.issues
        )

        if not record.is_valid:
            counts["rejected"] += 1
            continue
        if record.fasta_id in known:
            counts["already_present"] += 1
            continue

        rows.append(
            [
                source_file,
                record.fasta_id,
                record.header,
                record.uniprot_accession,
                record.sequence,
                record.sequence_length,
                now,
            ]
        )
        if len(rows) >= batch_size or (limit is not None and len(rows) >= limit):
            flush()
            if limit is not None and counts["inserted"] >= limit:
                break
    if rows or issues:
        flush()

    counts["elapsed"] = round(time.time() - started, 1)
    return counts


def compute_features(
    conn: duckdb.DuckDBPyConnection, *, batch_size: int = 1000
) -> dict[str, Any]:
    """Compute the characteristics of every protein that lacks them.

    A protein is pending when it has no feature row or when the stored
    ``feature_version`` differs from the current one. Proteins without a single
    canonical residue keep a row with NULL values, so they are not selected again.
    """
    started = time.time()
    counts = {"computed": 0, "skipped": 0}
    pending = """
        SELECT p.source_file, p.fasta_id, p.sequence
        FROM proteins p
        LEFT JOIN sequence_features f USING (source_file, fasta_id)
        WHERE f.fasta_id IS NULL OR f.feature_version <> ?
        LIMIT ?
    """
    feature_update = [c for c in FEATURE_TABLE_COLUMNS if c not in KEY]
    aac_update = [c for c in AAC_TABLE_COLUMNS if c not in KEY]
    dpc_update = [c for c in DPC_TABLE_COLUMNS if c not in KEY]

    while True:
        batch = conn.execute(
            pending, [features.FEATURE_VERSION, batch_size]
        ).fetchall()
        if not batch:
            break

        now = utc_now()
        feature_rows: list[list[Any]] = []
        aac_rows: list[list[Any]] = []
        dpc_rows: list[list[Any]] = []

        for source_file, fasta_id, sequence in batch:
            cleaned = features.clean_sequence(sequence)
            if not cleaned:
                feature_rows.append(
                    [
                        source_file,
                        fasta_id,
                        features.FEATURE_VERSION,
                        0,
                        len(sequence),
                        *([None] * 7),
                        now,
                    ]
                )
                counts["skipped"] += 1
                continue

            values = features.sequence_features(sequence)
            feature_rows.append(
                [
                    source_file,
                    fasta_id,
                    *[values[column] for column in features.FEATURE_COLUMNS],
                    now,
                ]
            )
            composition = features.amino_acid_composition(cleaned)
            aac_rows.append(
                [
                    source_file,
                    fasta_id,
                    *[composition[column] for column in features.AAC_COLUMNS],
                ]
            )
            pairs = features.dipeptide_composition(cleaned)
            dpc_rows.append(
                [
                    source_file,
                    fasta_id,
                    *[pairs[column] for column in features.DPC_COLUMNS],
                ]
            )
            counts["computed"] += 1

        with db.transaction(conn):
            db.write_rows(
                conn,
                "sequence_features",
                FEATURE_TABLE_COLUMNS,
                feature_rows,
                key=KEY,
                update=feature_update,
            )
            db.write_rows(
                conn,
                "amino_acid_composition",
                AAC_TABLE_COLUMNS,
                aac_rows,
                key=KEY,
                update=aac_update,
            )
            db.write_rows(
                conn,
                "dipeptide_composition",
                DPC_TABLE_COLUMNS,
                dpc_rows,
                key=KEY,
                update=dpc_update,
            )

    counts["elapsed"] = round(time.time() - started, 1)
    return counts


def enrich(
    conn: duckdb.DuckDBPyConnection,
    *,
    batch_size: int = 200,
    max_age_days: int | None = None,
    limit: int | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Retrieve UniProt metadata and GO annotations for pending proteins.

    A protein is pending when it has a UniProt accession and no annotation row,
    or when its row is older than ``max_age_days``. Each batch is committed in a
    single transaction, so an interrupted run loses nothing and the next run
    continues from the proteins still without an annotation.

    Raises:
        uniprot.UniProtError: If a request keeps failing. Nothing is recorded as
            missing for the accessions of the failed batch.
    """
    started = time.time()
    counts = {"queried": 0, "matched": 0, "not_found": 0, "go_rows": 0, "requests": 0}
    pending = """
        SELECT p.source_file, p.fasta_id, p.uniprot_accession
        FROM proteins p
        LEFT JOIN uniprot_annotations a USING (source_file, fasta_id)
        WHERE p.uniprot_accession IS NOT NULL
          AND (a.fasta_id IS NULL
               OR (? IS NOT NULL
                   AND a.retrieved_at < now() - CAST(? AS INTEGER) * INTERVAL 1 DAY))
        ORDER BY p.uniprot_accession, p.source_file, p.fasta_id
        LIMIT ?
    """
    client = session or requests.Session()

    while True:
        size = batch_size if limit is None else min(batch_size, limit - counts["queried"])
        if size <= 0:
            break

        batch = conn.execute(pending, [max_age_days, max_age_days, size]).fetchall()
        if not batch:
            break

        by_accession: dict[str, list[tuple[str, str]]] = {}
        for source_file, fasta_id, accession in batch:
            by_accession.setdefault(accession.upper(), []).append((source_file, fasta_id))

        found = uniprot.fetch(list(by_accession), session=client)
        counts["requests"] += 1
        counts["queried"] += len(batch)
        counts["matched"] += sum(
            len(keys) for accession, keys in by_accession.items() if accession in found
        )
        counts["not_found"] += sum(
            len(keys) for accession, keys in by_accession.items() if accession not in found
        )
        counts["go_rows"] += _store_annotations(conn, by_accession, found)

    counts["elapsed"] = round(time.time() - started, 1)
    return counts


def _store_annotations(
    conn: duckdb.DuckDBPyConnection,
    by_accession: dict[str, list[tuple[str, str]]],
    found: dict[str, uniprot.Annotation],
) -> int:
    """Persist one batch of results atomically; return the number of GO rows."""
    now = utc_now()
    annotation_rows: list[list[Any]] = []
    go_rows: list[list[Any]] = []

    for accession, keys in by_accession.items():
        annotation = found.get(accession)
        for source_file, fasta_id in keys:
            if annotation is None:
                annotation_rows.append(
                    [source_file, fasta_id, accession, *([None] * 11), "not_found", now]
                )
                continue
            annotation_rows.append(
                [
                    source_file,
                    fasta_id,
                    accession,
                    annotation.uniprot_id,
                    annotation.protein_name,
                    annotation.gene_names,
                    annotation.organism,
                    annotation.taxon_id,
                    annotation.protein_existence,
                    annotation.secondary_accessions,
                    annotation.keywords,
                    annotation.function_description,
                    annotation.subcellular_location,
                    annotation.source_length,
                    "matched",
                    now,
                ]
            )
            for go in annotation.go:
                go_rows.append(
                    [
                        source_file,
                        fasta_id,
                        go.go_id,
                        go.aspect,
                        go.go_name,
                        go.evidence_code,
                        go.evidence_source,
                        go.provenance,
                        now,
                    ]
                )

    annotation_update = [c for c in ANNOTATION_COLUMNS if c not in KEY]
    with db.transaction(conn):
        # GO rows are replaced per protein: the source may have withdrawn a term,
        # and keeping it would claim evidence that no longer exists.
        for source_file, fasta_ids in _group_ids(annotation_rows).items():
            placeholders = ", ".join("?" for _ in fasta_ids)
            conn.execute(
                "DELETE FROM protein_go_annotations "
                f"WHERE source_file = ? AND fasta_id IN ({placeholders})",
                [source_file, *fasta_ids],
            )
        db.write_rows(
            conn,
            "uniprot_annotations",
            ANNOTATION_COLUMNS,
            annotation_rows,
            key=KEY,
            update=annotation_update,
        )
        db.write_rows(conn, "protein_go_annotations", GO_COLUMNS, go_rows)

    return len(go_rows)


def _group_ids(rows: Sequence[Sequence[Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row[1])
    return grouped


def database_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Counts that describe what the database actually contains."""

    def scalar(sql: str) -> Any:
        return conn.execute(sql).fetchone()[0]

    return {
        "proteins": scalar("SELECT count(*) FROM proteins"),
        "without_accession": scalar(
            "SELECT count(*) FROM proteins WHERE uniprot_accession IS NULL"
        ),
        "with_features": scalar("SELECT count(*) FROM sequence_features"),
        "with_composition": scalar("SELECT count(*) FROM amino_acid_composition"),
        "with_dipeptides": scalar("SELECT count(*) FROM dipeptide_composition"),
        "feature_versions": scalar(
            "SELECT string_agg(DISTINCT CAST(feature_version AS VARCHAR), ',') "
            "FROM sequence_features"
        ),
        "matched": scalar(
            "SELECT count(*) FROM uniprot_annotations WHERE status = 'matched'"
        ),
        "not_found": scalar(
            "SELECT count(*) FROM uniprot_annotations WHERE status = 'not_found'"
        ),
        "unqueried": scalar(
            "SELECT count(*) FROM proteins p "
            "LEFT JOIN uniprot_annotations a USING (source_file, fasta_id) "
            "WHERE p.uniprot_accession IS NOT NULL AND a.fasta_id IS NULL"
        ),
        "go_rows": scalar("SELECT count(*) FROM protein_go_annotations"),
        "proteins_with_go": scalar(
            "SELECT count(DISTINCT (source_file || '|' || fasta_id)) "
            "FROM protein_go_annotations"
        ),
        "go_by_aspect": dict(
            conn.execute(
                "SELECT aspect, count(*) FROM protein_go_annotations "
                "GROUP BY aspect ORDER BY aspect"
            ).fetchall()
        ),
        "issues": dict(
            conn.execute(
                "SELECT issue, count(*) FROM ingestion_issues "
                "GROUP BY issue ORDER BY issue"
            ).fetchall()
        ),
        "orphans": scalar(
            "SELECT (SELECT count(*) FROM sequence_features f "
            "        LEFT JOIN proteins p USING (source_file, fasta_id) "
            "        WHERE p.fasta_id IS NULL) "
            "     + (SELECT count(*) FROM protein_go_annotations g "
            "        LEFT JOIN proteins p USING (source_file, fasta_id) "
            "        WHERE p.fasta_id IS NULL)"
        ),
    }
