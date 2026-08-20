"""Orchestrator for UniProt annotation enrichment with resume support."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from tqdm import tqdm

from protein_database.database.backfill import backfill_features
from protein_database.database.connection import bulk_upsert
from protein_database.uniprot.accession import group_accessions_by_local_rows
from protein_database.uniprot.batching import chunk_list
from protein_database.uniprot.cache import BatchCache
from protein_database.uniprot.client import UniProtClient
from protein_database.uniprot.parser import (
    STATUS_FETCHED,
    parse_tsv,
    missing_accessions,
    TsvParseError,
)

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_MISSING_FUNCTION = "missing_function"
STATUS_NOT_FOUND = "not_found"
STATUS_FAILED = "failed"

#: Columns of uniprot_annotations that are upserted.
ANNOTATION_COLUMNS = [
    "fasta_id",
    "source_file",
    "uniprot_accession",
    "uniprot_id",
    "protein_name",
    "gene_name",
    "function",
    "subcellular_location",
    "protein_existence",
    "organism",
    "keywords",
    "status",
    "retrieved_at",
]

ANNOTATION_UPDATE_COLUMNS = [
    "uniprot_accession",
    "uniprot_id",
    "protein_name",
    "gene_name",
    "function",
    "subcellular_location",
    "protein_existence",
    "organism",
    "keywords",
    "status",
    "retrieved_at",
]

QUEUE_COLUMNS = ["accession", "status", "retry_count", "last_attempt", "created_at", "completed_at"]


@dataclass
class RunSummary:
    """Summary report for a UniProt annotation run."""

    total_local_proteins: int = 0
    unique_accessions: int = 0
    successfully_retrieved: int = 0
    missing_function: int = 0
    not_found: int = 0
    failed: int = 0
    http_requests: int = 0
    cached_batches: int = 0
    function_coverage_pct: float = 0.0
    already_completed_skipped: int = 0


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_accession_queue(
    conn: duckdb.DuckDBPyConnection,
    accession_map: dict[str, list[tuple[str, str]]],
) -> int:
    """Populate the accession queue with pending accessions (idempotent).

    Args:
        conn: DuckDB connection.
        accession_map: Accession -> list of (fasta_id, source_file).

    Returns:
        Number of new accessions added.
    """
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT accession FROM uniprot_accession_queue"
        ).fetchall()
    }
    new_accessions = [acc for acc in accession_map if acc not in existing]
    if not new_accessions:
        return 0
    rows = [[acc, STATUS_PENDING, 0, None, _utcnow_str(), None] for acc in new_accessions]
    conn.executemany(
        "INSERT INTO uniprot_accession_queue "
        "(accession, status, retry_count, last_attempt, created_at, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def load_accession_map(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, list[tuple[str, str]]]:
    """Load the accession -> local protein rows mapping from the proteins table.

    Args:
        conn: DuckDB connection.

    Returns:
        Dict mapping UniProt accession to list of (fasta_id, source_file).
    """
    rows = conn.execute(
        "SELECT fasta_id, source_file, 'x' FROM proteins"
    ).fetchall()
    return group_accessions_by_local_rows(rows)


def upsert_annotations(
    conn: duckdb.DuckDBPyConnection,
    annotation_rows: list[dict[str, str | None]],
    accession_map: dict[str, list[tuple[str, str]]],
) -> int:
    """Upsert annotation rows into uniprot_annotations, fanning out to all
    local protein rows that share each accession.

    Args:
        conn: DuckDB connection.
        annotation_rows: Parsed annotation dicts (one per returned entry).
        accession_map: Accession -> local (fasta_id, source_file) rows.

    Returns:
        Number of annotation rows written.
    """
    write_rows: list[list] = []
    now = _utcnow_str()
    for ann in annotation_rows:
        acc = ann["uniprot_accession"]
        if not acc:
            continue
        local_rows = accession_map.get(acc, [])
        for fasta_id, source_file in local_rows:
            write_rows.append([
                fasta_id,
                source_file,
                acc,
                ann.get("uniprot_id"),
                ann.get("protein_name"),
                ann.get("gene_name"),
                ann.get("function"),
                ann.get("subcellular_location"),
                ann.get("protein_existence"),
                ann.get("organism"),
                ann.get("keywords"),
                ann.get("status", STATUS_FETCHED),
                now,
            ])
    if not write_rows:
        return 0
    bulk_upsert(
        conn,
        "uniprot_annotations",
        ANNOTATION_COLUMNS,
        write_rows,
        on_conflict_columns=["fasta_id", "source_file"],
        update_columns=ANNOTATION_UPDATE_COLUMNS,
    )
    return len(write_rows)


def update_queue_status(
    conn: duckdb.DuckDBPyConnection,
    accessions: list[str],
    status: str,
) -> None:
    """Set the status of a set of accessions in the queue.

    Args:
        conn: DuckDB connection.
        accessions: List of accessions to update.
        status: Target status.
    """
    if not accessions:
        return
    now = _utcnow_str()
    completed_at = now if status in (STATUS_COMPLETED, STATUS_MISSING_FUNCTION, STATUS_NOT_FOUND) else None
    rows = [[status, now, completed_at, acc] for acc in accessions]
    conn.executemany(
        f"UPDATE uniprot_accession_queue SET status = ?, last_attempt = ?, "
        f"completed_at = ? WHERE accession = ?",
        rows,
    )


def increment_retry(conn: duckdb.DuckDBPyConnection, accessions: list[str]) -> None:
    """Increment retry_count for accessions (used for failed batches)."""
    if not accessions:
        return
    now = _utcnow_str()
    rows = [[now, acc] for acc in accessions]
    conn.executemany(
        "UPDATE uniprot_accession_queue SET retry_count = retry_count + 1, "
        "last_attempt = ? WHERE accession = ?",
        rows,
    )


def get_pending_accessions(conn: duckdb.DuckDBPyConnection, limit: int | None) -> list[str]:
    """Return accessions in pending/failed status for processing."""
    sql = "SELECT accession FROM uniprot_accession_queue WHERE status IN (?, ?) ORDER BY accession"
    params = [STATUS_PENDING, STATUS_FAILED]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def annotate_database(
    db_path: str | Path = "proteins.duckdb",
    cache_dir: str | Path = "cache/uniprot",
    batch_size: int = 100,
    limit: int | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    workers: int = 4,
    client: UniProtClient | None = None,
) -> RunSummary:
    """Enrich the protein database with UniProt annotations.

    Batching and database writes happen in the coordinator; HTTP requests are
    performed concurrently across ``workers`` threads.

    Args:
        db_path: Path to the DuckDB database.
        cache_dir: Directory for raw response cache.
        batch_size: Accessions per request. Capped at the UniProt API limit
            of 100 (default 100).
        limit: Max accessions to process this run (for testing/resume).
        max_retries: Max retries per request.
        base_delay: Initial backoff delay.
        workers: Number of concurrent HTTP-fetch threads (default 4).
        client: Optional UniProtClient (injectable for tests).

    Returns:
        RunSummary with run statistics.
    """
    conn = duckdb.connect(str(db_path)) if db_path != ":memory:" else duckdb.connect(":memory:")
    try:
        from protein_database.database.connection import init_database

        print("Reading protein table...")
        init_database(conn)

        # Recompute locally-computable features/composition (sequence-derived).
        # UniProt cannot supply these, so we backfill them from stored sequences.
        print("Backfilling sequence features and amino-acid composition...")
        backfill_report = backfill_features(conn)
        print(
            f"Backfilled features for {backfill_report['features_written']:,} "
            f"proteins and composition for {backfill_report['compositions_written']:,} "
            f"proteins ({backfill_report['errors']} errors)."
        )

        print("Extracting unique UniProt accessions...")
        accession_map = load_accession_map(conn)
        total_local_proteins = sum(len(v) for v in accession_map.values())
        print(f"Found {total_local_proteins:,} local protein rows, {len(accession_map):,} unique accessions.")

        print("Building accession queue...")
        build_accession_queue(conn, accession_map)

        summary = RunSummary(
            total_local_proteins=total_local_proteins,
            unique_accessions=len(accession_map),
        )

        cache = BatchCache(cache_dir)
        client = client or UniProtClient(
            batch_size=batch_size,
            max_retries=max_retries,
            base_delay=base_delay,
        )

        pending = get_pending_accessions(conn, limit)
        print(f"{len(pending):,} accessions to process this run.")
        if not pending:
            print("Nothing to process — all accessions already handled.")
            return summary

        # Resume: reset in-flight accessions that never completed to pending.
        reset_inflight(conn, pending)

        batches = list(chunk_list(pending, client.batch_size))

        fetch_lock = threading.Lock()
        # requests.Session is not thread-safe; give each thread its own client.
        thread_local = threading.local()

        def _make_client() -> UniProtClient:
            if client is not None:
                # A shared injected client (e.g. in tests) — reuse it.
                return client
            actual_batch_size = client.batch_size if client else batch_size
            return UniProtClient(
                batch_size=actual_batch_size,
                max_retries=max_retries,
                base_delay=base_delay,
            )

        def _fetch(batch: list[str]) -> tuple[list[str], str | None]:
            """Fetch or load a batch, returning (batch, tsv_text).

            Cached batches are loaded from disk; otherwise a network request
            is made and the raw response is cached before returning.
            """
            batch_id = cache.batch_id(batch)
            if cache.exists(batch_id):
                return batch, cache.load(batch_id)
            thread_client = getattr(thread_local, "client", None)
            if thread_client is None:
                thread_client = _make_client()
                thread_local.client = thread_client
            try:
                resp = thread_client.fetch_batch(batch)
                text = resp.text
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch %s failed after retries: %s", batch_id, exc)
                return batch, None
            # Cache raw response BEFORE import so a crash can still resume.
            cache.save(batch_id, text)
            with fetch_lock:
                summary.http_requests += 1
            return batch, text

        progress_bar = tqdm(
            total=len(pending),
            desc="Annotating",
            unit="protein",
            leave=True,
        )
        lock = threading.Lock()
        workers = max(1, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch, b): b for b in batches}
            for future in as_completed(futures):
                batch, text = future.result()
                batch_id = cache.batch_id(batch)
                progress_bar.set_description(f"Annotating [{batch_id[:8]}]")

                if text is None:
                    # Fetch failed after retries.
                    with lock:
                        increment_retry(conn, batch)
                        update_queue_status(conn, batch, STATUS_FAILED)
                        summary.failed += len(batch)
                    progress_bar.update(len(batch))
                    progress_bar.set_postfix(**_postfix(summary))
                    continue

                # Parse + validate (don't mark not_found unless parse succeeds).
                try:
                    annotations = parse_tsv(text, batch)
                except TsvParseError as exc:
                    logger.error("Unparseable batch %s: %s", batch_id, exc)
                    with lock:
                        increment_retry(conn, batch)
                        update_queue_status(conn, batch, STATUS_FAILED)
                        summary.failed += len(batch)
                    progress_bar.update(len(batch))
                    progress_bar.set_postfix(**_postfix(summary))
                    continue

                # Determine not-found accessions ONLY after a confirmed parse.
                not_found = [acc for acc in missing_accessions(batch, annotations) if acc in batch]
                found = sorted(set(batch) - set(not_found))

                # Upsert (fan out to local rows) — coordinated, not concurrent.
                with lock:
                    written = upsert_annotations(conn, annotations, accession_map)
                    summary.successfully_retrieved += written
                    missing_fn = [a for a in annotations if a["status"] == STATUS_MISSING_FUNCTION]
                    summary.missing_function += len(missing_fn)
                    update_queue_status(conn, found, STATUS_COMPLETED)
                    if not_found:
                        update_queue_status(conn, not_found, STATUS_NOT_FOUND)
                        summary.not_found += len(not_found)

                progress_bar.update(len(batch))
                progress_bar.set_postfix(**_postfix(summary))

        progress_bar.close()

        # Function coverage.
        coverage = compute_function_coverage(conn)
        summary.function_coverage_pct = coverage
        summary.cached_batches = len(cache.list_batches())
        return summary
    finally:
        conn.close()


def _postfix(summary: RunSummary) -> dict:
    """Build the tqdm postfix dict from the run summary."""
    return {
        "retrieved": summary.successfully_retrieved,
        "missing_fn": summary.missing_function,
        "not_found": summary.not_found,
        "failed": summary.failed,
    }


def reset_inflight(conn: duckdb.DuckDBPyConnection, pending: list[str]) -> None:
    """No-op placeholder; resume handled by re-processing pending/failed.

    Kept for clarity: in-flight accessions that were left in a non-terminal
    state are already re-fetched in a new run because they are 'pending'.
    """
    del conn, pending


def compute_function_coverage(conn: duckdb.DuckDBPyConnection) -> float:
    """Compute the fraction of local proteins with a Function annotation.

    Args:
        conn: DuckDB connection.

    Returns:
        Percentage (0-100) of proteins with non-empty function.
    """
    total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
    if total == 0:
        return 0.0
    with_fn = conn.execute(
        "SELECT count(*) FROM uniprot_annotations "
        "WHERE function IS NOT NULL AND function != ''"
    ).fetchone()[0]
    return (with_fn / total) * 100.0