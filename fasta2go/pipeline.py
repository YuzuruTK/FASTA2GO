"""Extraction pipeline: one pass over the FASTA that fills every table.

The FASTA is read in batches. For every batch the sequences are analysed, the
accessions are queried on UniProt, the lines are consolidated into CSV and all
the rows are written in one transaction before the next batch starts, so a batch
is either completely stored or not stored at all.

A complete extraction starts from an empty database: nothing is looked up
beforehand and nothing is updated afterwards.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import requests

from fasta2go import db, features, uniprot
from fasta2go.fasta import read_fasta

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


def extract(
    conn: duckdb.DuckDBPyConnection,
    fasta_path: str | Path,
    *,
    batch_size: int = 5000,
    limit: int | None = None,
    accessions_per_request: int = 200,
    with_uniprot: bool = True,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Read the FASTA in batches and store each batch as a whole.

    Every batch goes through the complete flow: the records are read and
    validated, their features are computed, their accessions are queried on
    UniProt, the lines of the batch are consolidated into CSV and the rows of
    the seven tables are written in one transaction. The next batch starts only
    when the previous one is stored, so a failure leaves the database unchanged.

    The database is expected to be empty: no protein is looked up beforehand and
    nothing is updated afterwards. ``batch_size`` is how many proteins the batch
    holds in memory at once (the script reads it from ``fasta2go.config``),
    ``limit`` stops after N stored proteins and ``with_uniprot=False`` stores the
    proteins and their features only.

    Raises:
        uniprot.UniProtError: If a request keeps failing. The batch is not
            written at all, so no accession is recorded as missing.
    """
    started = time.time()
    path = Path(fasta_path)
    source_file = path.name
    client = session or (requests.Session() if with_uniprot else None)

    counts: dict[str, Any] = {
        "records": 0,
        "inserted": 0,
        "rejected": 0,
        "duplicates": 0,
        "computed": 0,
        "skipped": 0,
        "queried": 0,
        "matched": 0,
        "not_found": 0,
        "go_rows": 0,
        "requests": 0,
        "batches": 0,
    }
    protein_rows: list[list[Any]] = []
    issue_rows: list[list[Any]] = []
    feature_rows: list[list[Any]] = []
    aac_rows: list[list[Any]] = []
    dpc_rows: list[list[Any]] = []
    by_accession: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()

    def flush() -> None:
        """Query UniProt for the batch and write every table in one transaction."""
        nonlocal protein_rows, issue_rows, feature_rows, aac_rows, dpc_rows
        nonlocal by_accession

        annotation_rows: list[list[Any]] = []
        go_rows: list[list[Any]] = []
        if by_accession:
            found: dict[str, uniprot.Annotation] = {}
            accessions = list(by_accession)
            for start in range(0, len(accessions), accessions_per_request):
                chunk = [
                    accession
                    for accession in accessions[start : start + accessions_per_request]
                    if uniprot.ACCESSION_PATTERN.match(accession)
                ]
                if not chunk:
                    continue
                found.update(uniprot.fetch(chunk, session=client))
                counts["requests"] += 1
            counts["queried"] += sum(len(keys) for keys in by_accession.values())
            counts["matched"] += sum(
                len(keys)
                for accession, keys in by_accession.items()
                if accession in found
            )
            counts["not_found"] += sum(
                len(keys)
                for accession, keys in by_accession.items()
                if accession not in found
            )
            annotation_rows, go_rows = _annotation_rows(by_accession, found, utc_now())

        stored = len(protein_rows)
        with db.transaction(conn):
            db.write_rows(conn, "proteins", PROTEIN_COLUMNS, protein_rows)
            db.write_rows(conn, "sequence_features", FEATURE_TABLE_COLUMNS, feature_rows)
            db.write_rows(conn, "amino_acid_composition", AAC_TABLE_COLUMNS, aac_rows)
            db.write_rows(conn, "dipeptide_composition", DPC_TABLE_COLUMNS, dpc_rows)
            db.write_rows(conn, "uniprot_annotations", ANNOTATION_COLUMNS, annotation_rows)
            db.write_rows(conn, "protein_go_annotations", GO_COLUMNS, go_rows)
            db.write_rows(conn, "ingestion_issues", ISSUE_COLUMNS, issue_rows)

        counts["inserted"] += stored
        counts["go_rows"] += len(go_rows)
        if stored:
            counts["batches"] += 1
        protein_rows, issue_rows = [], []
        feature_rows, aac_rows, dpc_rows = [], [], []
        by_accession = {}

    processed = 0
    in_batch = 0
    for record in read_fasta(path):
        if limit is not None and processed >= limit:
            break
        counts["records"] += 1

        if record.fasta_id in seen:
            counts["duplicates"] += 1
            issue_rows.append(
                [
                    source_file,
                    record.line_number,
                    record.fasta_id,
                    "duplicate_id",
                    "kept the first occurrence",
                    utc_now(),
                ]
            )
            continue
        seen.add(record.fasta_id)
        issue_rows.extend(
            [source_file, record.line_number, record.fasta_id, code, detail, utc_now()]
            for code, detail in record.issues
        )

        if not record.is_valid:
            counts["rejected"] += 1
            continue

        now = utc_now()
        protein_rows.append(
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
        feature_row, aac_row, dpc_row, defined = _feature_rows(
            source_file, record.fasta_id, record.sequence, now
        )
        feature_rows.append(feature_row)
        aac_rows.append(aac_row)
        dpc_rows.append(dpc_row)
        counts["computed" if defined else "skipped"] += 1

        if with_uniprot and record.uniprot_accession is not None:
            by_accession.setdefault(record.uniprot_accession.upper(), []).append(
                (source_file, record.fasta_id)
            )

        processed += 1
        in_batch += 1
        if in_batch >= batch_size:
            flush()
            in_batch = 0

    if in_batch or issue_rows:
        flush()

    counts["elapsed"] = round(time.time() - started, 1)
    return counts


def _feature_rows(
    source_file: str, fasta_id: str, sequence: str, now: datetime
) -> tuple[list[Any], list[Any], list[Any], bool]:
    """Rows for the three feature tables and whether any formula was defined.

    AAC uses the canonical residues; DPC needs the original sequence so a
    non-canonical residue breaks adjacency instead of creating a new pair. A
    protein without a single canonical residue still gets one row in each table,
    with NULL values, so the three tables stay aligned with ``proteins``.
    """
    cleaned = features.clean_sequence(sequence)
    if cleaned:
        values = features.sequence_features(sequence)
    else:
        values = {
            "feature_version": features.FEATURE_VERSION,
            "canonical_length": 0,
            "noncanonical_residues": len(sequence),
            "canonical_pairs": 0,
        }

    feature_row = [
        source_file,
        fasta_id,
        *[values.get(column) for column in features.FEATURE_COLUMNS],
        now,
    ]
    composition = features.amino_acid_composition(cleaned)
    aac_row = [
        source_file,
        fasta_id,
        *[composition[column] for column in features.AAC_COLUMNS],
    ]
    pairs = features.dipeptide_composition(sequence)
    dpc_row = [
        source_file,
        fasta_id,
        *[pairs[column] for column in features.DPC_COLUMNS],
    ]
    return feature_row, aac_row, dpc_row, bool(cleaned)


def _annotation_rows(
    by_accession: dict[str, list[tuple[str, str]]],
    found: dict[str, uniprot.Annotation],
    now: datetime,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Rows for ``uniprot_annotations`` and ``protein_go_annotations``.

    An accession the source did not return is stored as ``not_found`` and gets no
    GO row: a missing answer never becomes the annotation of another protein.
    """
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
    return annotation_rows, go_rows


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

