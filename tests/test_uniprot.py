"""Tests for UniProt snapshot ingestion and the incremental API mode.

No test touches the network: HTTP is replaced by a fake session, so the
contract that matters - request, parse, upsert, failure handling - is checked
offline.
"""

import gzip
from pathlib import Path

import pytest

import protein_database.uniprot as uniprot
from protein_database.db import connect, init_database

HEADER = (
    "Entry\tEntry Name\tProtein names\tGene Names\tFunction [CC]\t"
    "Subcellular location [CC]\tProtein existence\tOrganism\tKeywords"
)

#: The snapshot additionally carries the GO columns, which TSV_MAP ignores.
GO_HEADER = "\t".join(
    [
        "Gene Ontology (biological process)",
        "Gene Ontology (cellular component)",
        "Gene Ontology (molecular function)",
        "Gene Ontology IDs",
    ]
)
HEADER_WITH_GO = HEADER + "\t" + GO_HEADER

ROWS = [
    "P11111\tAAA_TEST\tTest protein alpha\tGENA\tFUNCTION: first function.\t"
    "Cytoplasm\tEvidence at protein level\tHomo sapiens\tReference proteome",
    "P22222\tBBB_TEST\tTest protein beta\tGENB\t\t\t"
    "Inferred from homology\tMus musculus\tTransferase",
]

#: Same rows with the four GO columns appended (new snapshot shape).
ROWS_WITH_GO = [row + "\t\t\t\t" for row in ROWS]
API_ROWS = {"P11111": ROWS_WITH_GO[0], "P22222": ROWS_WITH_GO[1]}

#: Local proteins: two present in UniProt, one absent (MISSING1).
PROTEINS = [
    ("sp|P11111|AAA_TEST", "test.fasta", "alpha", "MKT"),
    ("sp|P22222|BBB_TEST", "test.fasta", "beta", "AAA"),
    ("sp|MISSING1|ZZZ_TEST", "test.fasta", "missing", "CCC"),
]


def add_proteins(conn, proteins=PROTEINS) -> None:
    """Insert protein rows through bound parameters (no SQL string building)."""
    conn.executemany(
        "INSERT INTO proteins (fasta_id, source_file, description, sequence, "
        "sequence_length, created_at) VALUES (?, ?, ?, ?, ?, now())",
        [(fasta_id, source, description, sequence, len(sequence))
         for fasta_id, source, description, sequence in proteins],
    )


def add_annotation(conn, fasta_id, status="fetched", source="test.fasta",
                   uniprot_id=None, function=None) -> None:
    conn.execute(
        "INSERT INTO uniprot_annotations (fasta_id, source_file, uniprot_id, "
        "function, status, retrieved_at) VALUES (?, ?, ?, ?, ?, now())",
        [fasta_id, source, uniprot_id, function, status],
    )


def tsv(rows) -> str:
    """An API-style TSV body with the 13-column header."""
    return HEADER_WITH_GO + "\n" + "\n".join(rows) + "\n"


@pytest.fixture
def protein_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "uniprot.duckdb"
    conn = connect(db_path)
    init_database(conn)
    add_proteins(conn)
    conn.close()
    return db_path


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(HEADER + "\n" + "\n".join(ROWS) + "\n")
    return path


class FakeResponse:
    """Minimal stand-in for a requests response."""

    def __init__(self, status_code: int, text: str, headers=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class FakeSession:
    """Replays responses and records which accessions each request asked for."""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def get(self, url: str, params=None, timeout=None) -> FakeResponse:
        params = params or {}
        self.calls.append(str(params.get("accessions", "")).split(","))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        if callable(self.responses[0]):
            return self.responses[0](params)
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Snapshot mode
# ---------------------------------------------------------------------------


def test_download_snapshot_reuses_existing(tmp_path: Path) -> None:
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    dest = bulk / uniprot.SNAPSHOT_NAME
    dest.write_bytes(b"cached")
    assert uniprot.download_snapshot(bulk) == dest


def test_staging_accepts_go_columns(tmp_path: Path) -> None:
    """The current 13-column snapshot must stage (regression: column count)."""
    path = tmp_path / "with_go.tsv"
    path.write_text(tsv(ROWS_WITH_GO))

    conn = connect(":memory:")
    init_database(conn)
    try:
        assert uniprot._build_staging_table(conn, [str(path)]) == 2
        staged = conn.execute(f"SELECT count(*) FROM {uniprot.STAGING_TABLE}").fetchone()
        assert staged[0] == 2
    finally:
        conn.close()


def test_staging_rejects_missing_mapped_column(tmp_path: Path) -> None:
    path = tmp_path / "broken.tsv"
    path.write_text("Entry\tEntry Name\nP11111\tAAA\n")

    conn = connect(":memory:")
    init_database(conn)
    try:
        with pytest.raises(ValueError, match="missing columns"):
            uniprot._build_staging_table(conn, [str(path)])
    finally:
        conn.close()


def test_bulk_ingest_annotates_matching_proteins(
    protein_db: Path, snapshot: Path, monkeypatch
) -> None:
    monkeypatch.setattr(uniprot, "download_snapshot", lambda *a, **k: snapshot)

    summary = uniprot.annotate_database(db_path=protein_db, source="bulk")

    assert summary["total_local_proteins"] == 3
    assert summary["annotated"] == 2
    assert summary["missing_function"] == 1
    assert summary["not_found"] == 1

    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT uniprot_id, function, organism, status FROM uniprot_annotations "
            "WHERE fasta_id = ?",
            ["sp|P11111|AAA_TEST"],
        ).fetchone() == (
            "AAA_TEST",
            "FUNCTION: first function.",
            "Homo sapiens",
            "fetched",
        )
        assert conn.execute(
            "SELECT status FROM uniprot_annotations WHERE fasta_id = ?",
            ["sp|MISSING1|ZZZ_TEST"],
        ).fetchone() == ("not_found",)
    finally:
        conn.close()


def test_bulk_fans_out_to_duplicate_rows(
    protein_db: Path, snapshot: Path, monkeypatch
) -> None:
    conn = connect(protein_db)
    add_proteins(conn, [("sp|P11111|AAA_TEST", "other.fasta", "alpha2", "MKT")])
    conn.close()
    monkeypatch.setattr(uniprot, "download_snapshot", lambda *a, **k: snapshot)

    summary = uniprot.annotate_database(db_path=protein_db, source="bulk")
    assert summary["annotated"] == 3

    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM uniprot_annotations WHERE fasta_id = ?",
            ["sp|P11111|AAA_TEST"],
        ).fetchone()[0] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_stale_rows_filters(protein_db: Path) -> None:
    conn = connect(protein_db)
    try:
        assert [row[2] for row in uniprot.select_stale_rows(conn)] == [
            "MISSING1",
            "P11111",
            "P22222",
        ]
        assert len(uniprot.select_stale_rows(conn, only_missing=True)) == 3

        add_annotation(conn, "sp|P11111|AAA_TEST")
        assert [row[2] for row in uniprot.select_stale_rows(conn, only_missing=True)] == [
            "MISSING1",
            "P22222",
        ]
        assert [row[2] for row in uniprot.select_stale_rows(conn, older_than_days=1)] == [
            "MISSING1",
            "P22222",
        ]
        assert len(uniprot.select_stale_rows(conn, only_missing=True, limit=1)) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def test_request_tsv_returns_the_body(monkeypatch) -> None:
    session = FakeSession([FakeResponse(200, tsv([ROWS_WITH_GO[0]]))])

    text = uniprot._request_tsv(
        ["P11111"], session, max_retries=0, base_delay=1.0, max_delay=30.0
    )

    assert text.startswith("Entry")
    assert session.calls == [["P11111"]]


def test_request_tsv_retries_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: sleeps.append(seconds))
    session = FakeSession(
        [
            FakeResponse(429, "slow down", {"Retry-After": "2"}),
            FakeResponse(200, tsv([ROWS_WITH_GO[0]])),
        ]
    )

    text = uniprot._request_tsv(
        ["P11111"], session, max_retries=1, base_delay=1.0, max_delay=30.0
    )

    assert "P11111" in text
    assert sleeps == [2.0]
    assert len(session.calls) == 2


def test_empty_body_raises_instead_of_meaning_not_found(monkeypatch) -> None:
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)
    session = FakeSession([FakeResponse(200, "")] * 2)

    with pytest.raises(uniprot.UniProtRequestError, match="empty body"):
        uniprot._request_tsv(
            ["P11111"], session, max_retries=1, base_delay=0.1, max_delay=1.0
        )
    assert len(session.calls) == 2


def test_http_error_raises_request_error(monkeypatch) -> None:
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)
    session = FakeSession([FakeResponse(403, "forbidden")])

    with pytest.raises(uniprot.UniProtRequestError, match="HTTP 403"):
        uniprot._request_tsv(
            ["P11111"], session, max_retries=2, base_delay=0.1, max_delay=1.0
        )
    assert len(session.calls) == 1


def test_fetch_tsv_header_only_means_no_entry() -> None:
    session = FakeSession([FakeResponse(200, HEADER_WITH_GO + "\n")])

    header, rows, rejected = uniprot.fetch_tsv(["P99999"], session=session)

    assert header == HEADER_WITH_GO
    assert rows == [] and rejected == []


def test_fetch_tsv_rejects_malformed_without_request() -> None:
    session = FakeSession([])

    header, rows, rejected = uniprot.fetch_tsv(["not an accession"], session=session)

    assert header is None and rows == []
    assert rejected == ["not an accession"]
    assert session.calls == []


def test_fetch_tsv_isolates_one_rejected_accession(monkeypatch) -> None:
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)
    rejected_id = "A0A0000000"

    def responder(params):
        requested = params["accessions"].split(",")
        if rejected_id in requested:
            return FakeResponse(400, "Accession " + rejected_id + " has invalid format")
        return FakeResponse(200, tsv([API_ROWS[a] for a in requested if a in API_ROWS]))

    session = FakeSession([responder])
    header, rows, rejected = uniprot.fetch_tsv(
        ["P11111", rejected_id, "P22222"], session=session, max_retries=0
    )

    assert header == HEADER_WITH_GO
    assert rejected == [rejected_id]
    assert len(rows) == 2
    assert len(session.calls) == 4


# ---------------------------------------------------------------------------
# API mode end to end (fake HTTP, real DuckDB)
# ---------------------------------------------------------------------------


def fetch_only(rows: dict[str, str]):
    """fetch_tsv replacement answering only the requested accessions."""

    def fetch(accessions, **kwargs):
        return HEADER_WITH_GO, [rows[a] for a in accessions if a in rows], []

    return fetch


def test_api_writes_rows_and_marks_not_found(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(uniprot, "fetch_tsv", fetch_only(API_ROWS))

    summary = uniprot.annotate_database(
        db_path=protein_db,
        cache_dir=tmp_path / "cache",
        source="api",
        only_missing=True,
    )

    assert summary["mode"] == "api"
    assert summary["accessions"] == 3
    assert summary["annotated"] == 2
    assert summary["missing_function"] == 1
    assert summary["not_found"] == 1
    assert summary["http_requests"] == 1
    assert summary["failed"] == 0

    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT uniprot_id, function, status FROM uniprot_annotations "
            "WHERE fasta_id = ?",
            ["sp|P11111|AAA_TEST"],
        ).fetchone() == ("AAA_TEST", "FUNCTION: first function.", "fetched")
        assert conn.execute(
            "SELECT status FROM uniprot_annotations WHERE fasta_id = ?",
            ["sp|MISSING1|ZZZ_TEST"],
        ).fetchone() == ("not_found",)
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 3
    finally:
        conn.close()


def test_api_requests_are_chunked(protein_db: Path, tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    fetcher = fetch_only(API_ROWS)

    def fetch(accessions, **kwargs):
        calls.append(list(accessions))
        return fetcher(accessions, **kwargs)

    monkeypatch.setattr(uniprot, "fetch_tsv", fetch)
    summary = uniprot.annotate_database(
        db_path=protein_db,
        cache_dir=tmp_path / "cache",
        source="api",
        only_missing=True,
        chunk_size=2,
    )

    assert sorted(len(call) for call in calls) == [1, 2]
    assert summary["http_requests"] == 2
    assert summary["annotated"] == 2


def test_transport_failure_is_not_recorded_as_not_found(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    session = FakeSession([FakeResponse(200, "")] * 10)
    monkeypatch.setattr(uniprot.requests, "Session", lambda: session)
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)

    summary = uniprot.annotate_database(
        db_path=protein_db,
        cache_dir=tmp_path / "cache",
        source="api",
        only_missing=True,
        max_retries=1,
    )

    assert summary["failed"] == 3
    assert summary["annotated"] == 0
    assert summary["not_found"] == 0

    conn = connect(protein_db)
    try:
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 0
    finally:
        conn.close()


def test_http_failure_keeps_existing_annotations(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    conn = connect(protein_db)
    add_annotation(
        conn,
        "sp|P11111|AAA_TEST",
        uniprot_id="AAA_OLD",
        function="FUNCTION: previous run.",
    )
    conn.close()

    session = FakeSession([FakeResponse(503, "maintenance")] * 10)
    monkeypatch.setattr(uniprot.requests, "Session", lambda: session)
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)

    summary = uniprot.annotate_database(
        db_path=protein_db,
        cache_dir=tmp_path / "cache",
        source="api",
        older_than_days=0,
        max_retries=0,
    )

    assert summary["failed"] == 3
    assert summary["not_found"] == 0

    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT uniprot_id, function, status FROM uniprot_annotations "
            "WHERE fasta_id = ?",
            ["sp|P11111|AAA_TEST"],
        ).fetchone() == ("AAA_OLD", "FUNCTION: previous run.", "fetched")
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 1
    finally:
        conn.close()


def test_nothing_to_do_makes_no_request(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    conn = connect(protein_db)
    for fasta_id, _, _, _ in PROTEINS:
        add_annotation(conn, fasta_id)
    conn.close()

    def forbidden():
        raise AssertionError("no HTTP session should be created")

    monkeypatch.setattr(uniprot.requests, "Session", forbidden)

    summary = uniprot.annotate_database(
        db_path=protein_db, cache_dir=tmp_path / "cache", source="api", only_missing=True
    )

    assert summary["accessions"] == 0
    assert summary["http_requests"] == 0
    assert summary["failed"] == 0


def test_filters_require_api_source(protein_db: Path) -> None:
    with pytest.raises(ValueError, match="require source"):
        uniprot.annotate_database(db_path=protein_db, source="bulk", only_missing=True)
    with pytest.raises(ValueError, match="chunk_size"):
        uniprot.annotate_database(db_path=protein_db, source="api", chunk_size=1001)
    with pytest.raises(ValueError, match="Unknown source"):
        uniprot.annotate_database(db_path=protein_db, source="nope")
