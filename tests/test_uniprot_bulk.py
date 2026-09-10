"""Tests for the bulk snapshot download and native TSV ingestion."""

import gzip
from pathlib import Path

import pytest
import requests

from protein_database.database.connection import connect, init_database
from protein_database.uniprot.bulk import (
    REVIEWED_QUERY,
    download_chunk,
    download_snapshot,
    split_query,
)
from protein_database.uniprot.client import STREAM_URL, UNIPROT_FIELDS, UniProtClient
from protein_database.uniprot.ingest import (
    build_staging_table,
    bulk_annotate_database,
    count_missing_accessions,
    count_unique_local_accessions,
    expand_sources,
    upsert_staged_annotations,
)

HEADER = "Entry\tEntry Name\tProtein names\tGene Names\tFunction [CC]\tSubcellular location [CC]\tProtein existence\tOrganism\tKeywords"

BATCH_ROWS = [
    "P11111\tAAA_TEST\tTest protein alpha\tGENA\tFUNCTION: first function.\tCytoplasm\tEvidence at protein level\tHomo sapiens\tReference proteome",
    "P22222\tBBB_TEST\tTest protein beta\tGENB\t\t\tInferred from homology\tMus musculus\tTransferase",
]

BULK_ROWS = [
    "P33333\tCCC_TEST\tTest protein gamma\tGENC\tFUNCTION: third function.\tNucleus\tEvidence at transcript level\tRattus norvegicus\tKinase",
    # Duplicate accession also present in the legacy cache (bulk must win).
    "P11111\tAAA_TEST\tTest protein alpha UPDATED\tGENA\tFUNCTION: newer function.\tCytoplasm\tEvidence at protein level\tHomo sapiens\tReference proteome",
]


@pytest.fixture
def protein_db(tmp_path: Path) -> Path:
    """Create a small database with three local protein rows."""
    db_path = tmp_path / "bulk.duckdb"
    conn = connect(db_path)
    init_database(conn)
    conn.execute(
        "INSERT INTO proteins (fasta_id, source_file, description, sequence, sequence_length, created_at) "
        "VALUES "
        "('sp|P11111|AAA_TEST', 'test.fasta', 'alpha', 'MKT', 3, now()), "
        "('sp|P22222|BBB_TEST', 'test.fasta', 'beta', 'AAA', 3, now()), "
        "('sp|MISSING1|ZZZ_TEST', 'test.fasta', 'missing', 'CCC', 3, now())"
    )
    conn.close()
    return db_path


@pytest.fixture
def tsv_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Write a legacy batch TSV and a gzipped bulk chunk."""
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    batch_file = batch_dir / "aaaa.tsv"
    batch_file.write_text(HEADER + "\n" + "\n".join(BATCH_ROWS) + "\n", encoding="utf-8")
    # A duplicate row in a second batch file (retried-batch overlap).
    dup_file = batch_dir / "bbbb.tsv"
    dup_file.write_text(HEADER + "\n" + BATCH_ROWS[0] + "\n", encoding="utf-8")

    bulk_dir = tmp_path / "bulk"
    bulk_dir.mkdir()
    bulk_file = bulk_dir / "chunk.tsv.gz"
    with gzip.open(bulk_file, "wt", encoding="utf-8") as fh:
        fh.write(HEADER + "\n" + "\n".join(BULK_ROWS) + "\n")
    return bulk_dir, batch_dir


def test_split_query_respects_or_limit() -> None:
    terms = [f"P{i:05d}" for i in range(250)]
    queries = split_query(terms)
    assert len(queries) == 3
    assert queries[0].count(" OR ") == 99
    assert all(q.startswith("accession:") for q in queries)
    assert split_query([]) == []


def test_download_snapshot_reuses_manifest(tmp_path: Path) -> None:
    """An existing manifest short-circuits any network access."""
    out_dir = tmp_path / "bulk"

    class FakeClient:
        def fetch_stream(self, query, dest):
            Path(dest).write_bytes(b"fake")
            return Path(dest)

    created = download_snapshot(["P1"], dest_dir=out_dir, client=FakeClient())

    class ExplodingClient:
        def fetch_stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("no network expected")

    assert download_snapshot(["P1"], dest_dir=out_dir, client=ExplodingClient()) == created


def test_download_snapshot_refresh_redownloads(tmp_path: Path) -> None:
    out_dir = tmp_path / "bulk"
    out_dir.mkdir()
    (out_dir / "chunk.tsv.gz").write_bytes(b"fake")
    (out_dir / "deadbeef.sha1").write_text("chunk.tsv.gz\n", encoding="utf-8")

    class FakeClient:
        def fetch_stream(self, query, dest):
            Path(dest).write_bytes(b"fresh")
            return Path(dest)

    chunks = download_snapshot(["P1"], dest_dir=out_dir, refresh=True, client=FakeClient())
    assert len(chunks) == 1
    assert chunks[0].read_bytes() == b"fresh"


def test_download_chunk_atomic_rename(tmp_path: Path) -> None:
    dest = tmp_path / "chunk.tsv.gz"

    class FakeClient:
        def fetch_stream(self, query, part):
            Path(part).write_bytes(b"data")
            return Path(part)

    download_chunk(FakeClient(), "accession:P1", dest, "fp")
    assert dest.read_bytes() == b"data"
    assert not list(tmp_path.glob("*.part"))
    assert (tmp_path / "fp.sha1").read_text(encoding="utf-8") == "chunk.tsv.gz\n"


def test_download_snapshot_downloads_and_reuses(tmp_path: Path) -> None:
    out_dir = tmp_path / "bulk"

    class FakeClient:
        def fetch_stream(self, query, dest):
            Path(dest).write_bytes(b"data:" + query.encode())
            return Path(dest)

    first = download_snapshot(["P1"], dest_dir=out_dir, client=FakeClient())
    assert len(first) == 1 and first[0].exists()

    class ExplodingClient:
        def fetch_stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must reuse snapshot")

    assert download_snapshot(["P1"], dest_dir=out_dir, client=ExplodingClient()) == first


def test_fetch_stream_params_and_retry() -> None:
    """Stream request uses reviewed query params and retries transient errors."""

    class FakeResp:
        status_code = 200

        def iter_content(self, chunk_size):
            yield b"gz-bytes"

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, stream=None, timeout=None):
            self.calls.append((url, params, stream))
            if len(self.calls) == 1:
                raise requests.ConnectionError("boom")
            return FakeResp()

    session = FakeSession()
    client = UniProtClient(max_retries=2, base_delay=0.0, session=session)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "snap.tsv.gz"
        client.fetch_stream(REVIEWED_QUERY, dest)
        assert dest.read_bytes() == b"gz-bytes"
    assert len(session.calls) == 2
    url, params, stream = session.calls[0]
    assert url == STREAM_URL
    assert params["query"] == REVIEWED_QUERY
    assert params["format"] == "tsv"
    assert params["fields"] == UNIPROT_FIELDS
    assert params["compressed"] == "true"
    assert stream is True


def test_expand_sources_order_and_dedup(tmp_path: Path, tsv_sources) -> None:
    bulk_dir, batch_dir = tsv_sources
    bulk_file = next(bulk_dir.glob("*.tsv.gz"))
    files = expand_sources([bulk_file, str(batch_dir / "*.tsv")])
    assert files[0] == bulk_file
    assert len(files) == 3
    # Missing globs contribute nothing.
    assert expand_sources([str(tmp_path / "nope" / "*.tsv")]) == []


def test_build_staging_table_dedupes_and_prefers_bulk(tmp_path: Path, tsv_sources) -> None:
    bulk_dir, batch_dir = tsv_sources
    conn = connect(":memory:")
    try:
        staged = build_staging_table(
            conn, [next(bulk_dir.glob("*.tsv.gz")), str(batch_dir / "*.tsv")]
        )
        # P11111 appears in bulk, batch, and the duplicate batch file.
        assert staged == 3
        name = conn.execute(
            'SELECT "Protein names" FROM uniprot_staging WHERE "Entry" = \'P11111\''
        ).fetchone()[0]
        assert name == "Test protein alpha UPDATED"
    finally:
        conn.close()


def test_build_staging_table_rejects_empty() -> None:
    conn = connect(":memory:")
    try:
        with pytest.raises(ValueError):
            build_staging_table(conn, ["/nonexistent/*.tsv"])
    finally:
        conn.close()


def test_upsert_staged_annotations_fans_out(protein_db: Path, tmp_path: Path, tsv_sources) -> None:
    bulk_dir, batch_dir = tsv_sources
    # Same accession in two local rows: both must be annotated.
    conn = connect(protein_db)
    try:
        conn.execute(
            "INSERT INTO proteins (fasta_id, source_file, description, sequence, sequence_length, created_at) "
            "VALUES ('sp|P11111|AAA_TEST', 'other.fasta', 'alpha2', 'MKT', 3, now())"
        )
        build_staging_table(conn, [next(bulk_dir.glob("*.tsv.gz")), str(batch_dir / "*.tsv")])
        written, missing_fn = upsert_staged_annotations(conn)
        assert written == 3
        # P22222 has an empty function -> missing_function; P33333 has no local row.
        assert missing_fn == 1
        row = conn.execute(
            "SELECT uniprot_id, function, organism, status FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST' AND source_file = 'test.fasta'"
        ).fetchone()
        assert row == ("AAA_TEST", "FUNCTION: newer function.", "Homo sapiens", "fetched")
        other = conn.execute(
            "SELECT protein_name FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST' AND source_file = 'other.fasta'"
        ).fetchone()
        assert other == ("Test protein alpha UPDATED",)
        assert count_missing_accessions(conn) == 1  # MISSING1 has no staged row
        assert count_unique_local_accessions(conn) == 3
    finally:
        conn.close()


def test_upsert_staged_annotations_overwrites(protein_db: Path, tmp_path: Path, tsv_sources) -> None:
    bulk_dir, batch_dir = tsv_sources
    conn = connect(protein_db)
    try:
        conn.execute(
            "INSERT INTO uniprot_annotations "
            "(fasta_id, source_file, uniprot_accession, uniprot_id, protein_name, gene_name, "
            "function, subcellular_location, protein_existence, organism, keywords, status, retrieved_at) "
            "VALUES ('sp|P11111|AAA_TEST', 'test.fasta', 'P11111', 'OLD', 'old', 'OLD', "
            "'old', 'old', 'old', 'old', 'old', 'fetched', now())"
        )
        build_staging_table(conn, [next(bulk_dir.glob("*.tsv.gz")), str(batch_dir / "*.tsv")])
        upsert_staged_annotations(conn)
        row = conn.execute(
            "SELECT uniprot_id, protein_name FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST' AND source_file = 'test.fasta'"
        ).fetchone()
        assert row == ("AAA_TEST", "Test protein alpha UPDATED")
    finally:
        conn.close()


def test_bulk_annotate_database_end_to_end(protein_db: Path, tsv_sources) -> None:
    bulk_dir, batch_dir = tsv_sources
    summary = bulk_annotate_database(
        db_path=protein_db,
        cache_dir=batch_dir,
        source="cache",
    )
    assert summary.total_local_proteins == 3
    assert summary.unique_accessions == 3
    assert summary.successfully_retrieved == 2
    assert summary.missing_function == 1
    assert summary.not_found == 1
    assert summary.http_requests == 0
    assert summary.cached_batches == 2
    assert summary.function_coverage_pct == pytest.approx(100 * 1 / 3)

    conn = connect(protein_db)
    try:
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 2
    finally:
        conn.close()


def test_bulk_annotate_database_rejects_unknown_source(protein_db: Path, tsv_sources) -> None:
    _, batch_dir = tsv_sources
    with pytest.raises(ValueError):
        bulk_annotate_database(db_path=protein_db, cache_dir=batch_dir, source="api")
