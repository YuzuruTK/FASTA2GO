"""End-to-end tests for the UniProt annotation pipeline."""

from pathlib import Path

import duckdb
import pytest

from protein_database.database.connection import connect, init_database
from protein_database.uniprot.annotate import annotate_database
from protein_database.uniprot.client import HttpResponse

HEADER = (
    "Entry\tEntry Name\tProtein names\tGene Names\tFunction [CC]\t"
    "Subcellular location [CC]\tProtein existence\tOrganism\tKeywords"
)


def _tsv(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


class StubClient:
    """A stub UniProtClient that serves scripted responses per batch.

    accessions_to_omit: accessions to NOT return in the response (not found).
    no_function_accessions: accessions returned with an empty Function.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        batch_size=200,
        accessions_to_omit: set[str] | None = None,
        no_function_accessions: set[str] | None = None,
    ):
        self.responses = responses or {}
        self.batch_size = batch_size
        self.calls = []
        self.accessions_to_omit = accessions_to_omit or set()
        self.no_function_accessions = no_function_accessions or set()

    def fetch_batch(self, accessions):
        import hashlib

        key = ",".join(sorted(accessions))
        batch_id = hashlib.sha1(key.encode()).hexdigest()[:16]
        self.calls.append(batch_id)
        if batch_id in self.responses:
            return HttpResponse(200, self.responses[batch_id])
        rows = []
        for acc in accessions:
            if acc in self.accessions_to_omit:
                continue
            if acc in self.no_function_accessions:
                rows.append(
                    f"{acc}\t{acc}_ID\tProtein {acc}\tGENE\t\t"
                    f"\tPredicted\tOrganism\tKW"
                )
            else:
                rows.append(
                    f"{acc}\t{acc}_ID\tProtein {acc}\tGENE\tFUNCTION: works\t"
                    f"\tEvidence\tOrganism\tKW"
                )
        return HttpResponse(200, _tsv(*rows))


def _make_db(tmp_path: Path, proteins: list[tuple[str, str]]) -> Path:
    db_path = tmp_path / "test.duckdb"
    conn = connect(db_path)
    init_database(conn)
    for fid, src in proteins:
        conn.execute(
            "INSERT INTO proteins (fasta_id, source_file, sequence, sequence_length, created_at) "
            "VALUES (?, ?, 'AAAA', 4, now())",
            [fid, src],
        )
    conn.close()
    return db_path


def _row(conn, fid, src):
    return conn.execute(
        "SELECT uniprot_accession, uniprot_id, function, status, protein_name "
        "FROM uniprot_annotations WHERE fasta_id = ? AND source_file = ?",
        [fid, src],
    ).fetchone()


def test_annotate_populates(tmp_path: Path) -> None:
    db = _make_db(tmp_path, [("sp|P69905|HBA_HUMAN", "a.fasta")])
    summary = annotate_database(
        db_path=db,
        cache_dir=tmp_path / "cache",
        batch_size=200,
        client=StubClient({}),
    )
    conn = connect(db)
    try:
        row = _row(conn, "sp|P69905|HBA_HUMAN", "a.fasta")
        assert row[0] == "P69905"
        assert row[1] == "P69905_ID"
        assert "FUNCTION: works" in row[2]
        assert row[3] == "fetched"
    finally:
        conn.close()
    assert summary.successfully_retrieved > 0


def test_one_accession_multiple_rows(tmp_path: Path) -> None:
    """One accession fanned out to two local rows, fetched once."""
    db = _make_db(
        tmp_path,
        [
            ("sp|P69905|HBA_HUMAN", "a.fasta"),
            ("sp|P69905|HBA_HUMAN", "b.fasta"),
        ],
    )
    client = StubClient({})
    summary = annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    conn = connect(db)
    try:
        r1 = _row(conn, "sp|P69905|HBA_HUMAN", "a.fasta")
        r2 = _row(conn, "sp|P69905|HBA_HUMAN", "b.fasta")
        assert r1[0] == "P69905"
        assert r2[0] == "P69905"
    finally:
        conn.close()
    assert len(client.calls) == 1  # one retrieval


def test_missing_function_status(tmp_path: Path) -> None:
    """Empty Function is stored as missing_function, other fields kept."""
    db = _make_db(tmp_path, [("tr|A0A0C5B5G6|MOTSC", "a.fasta")])
    # Response for that accession has no function
    client = StubClient(no_function_accessions={"A0A0C5B5G6"})
    summary = annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    conn = connect(db)
    try:
        row = _row(conn, "tr|A0A0C5B5G6|MOTSC", "a.fasta")
        assert row[3] == "missing_function"
        # protein_name preserved even without function
        assert row[4] is not None
    finally:
        conn.close()
    assert summary.missing_function >= 1


def test_not_found_marked(tmp_path: Path) -> None:
    """Accessions not returned by API are marked not_found."""
    db = _make_db(tmp_path, [("sp|P69905|HBA_HUMAN", "a.fasta"), ("sp|FAKE99|FAKE", "a.fasta")])
    # Stub returns only P69905 (omits FAKE99)
    client = StubClient(accessions_to_omit={"FAKE99"})
    summary = annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    conn = connect(db)
    try:
        # P69905 has annotation, FAKE99 is not found (no row written)
        row = _row(conn, "sp|FAKE99|FAKE", "a.fasta")
        assert row is None
    finally:
        conn.close()
    assert summary.not_found >= 1


def test_resume_from_cache(tmp_path: Path) -> None:
    """A second run reuses cached raw response without API call."""
    db = _make_db(tmp_path, [("sp|P69905|HBA_HUMAN", "a.fasta")])
    client = StubClient({})
    first = annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    calls_after_first = len(client.calls)

    # Second run: queue is completed, so no pending -> no new calls, no upserts
    second = annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    assert len(client.calls) == calls_after_first
    assert second.successfully_retrieved == 0


def test_dedupe_accessions(tmp_path: Path) -> None:
    """Duplicate accession entries produce one queue row and one fetch."""
    db = _make_db(
        tmp_path,
        [
            ("sp|P69905|HBA_HUMAN", "a.fasta"),
            ("sp|P69905|HBA_HUMAN", "b.fasta"),
        ],
    )
    client = StubClient({})
    annotate_database(db_path=db, cache_dir=tmp_path / "cache", client=client)
    conn = connect(db)
    try:
        count = conn.execute(
            "SELECT count(*) FROM uniprot_accession_queue WHERE accession = 'P69905'"
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_concurrent_multi_batch(tmp_path: Path) -> None:
    """Multiple batches are fetched concurrently and all upserted."""
    # 5 distinct accessions -> with batch_size 2 => 3 batches
    db = _make_db(
        tmp_path,
        [
            ("sp|P69905|HBA_HUMAN", "a.fasta"),
            ("sp|P68871|HBB_HUMAN", "a.fasta"),
            ("sp|P00740|FA9_HUMAN", "a.fasta"),
            ("sp|P01009|A1AT_HUMAN", "a.fasta"),
            ("sp|P0DTC2|SPIKE_SARS2", "a.fasta"),
        ],
    )
    client = StubClient(batch_size=2)
    summary = annotate_database(
        db_path=db,
        cache_dir=tmp_path / "cache",
        batch_size=2,
        workers=4,
        client=client,
    )
    conn = connect(db)
    try:
        n = conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0]
        assert n == 5
        # all marked completed (fetched)
        done = conn.execute(
            "SELECT count(*) FROM uniprot_accession_queue WHERE status = 'completed'"
        ).fetchone()[0]
        assert done == 5
    finally:
        conn.close()
    assert summary.successfully_retrieved == 5
    assert len(client.calls) == 3

