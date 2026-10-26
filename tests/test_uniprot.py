"""Tests for UniProt snapshot download and native annotation ingestion."""

import gzip
from pathlib import Path

import pytest
import requests

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
ROWS_WITH_GO = [
    row + "\t\t\t\t" for row in ROWS
]

PROTEINS_SQL = (
    "INSERT INTO proteins (fasta_id, source_file, description, sequence, "
    "sequence_length, created_at) VALUES "
    "('sp|P11111|AAA_TEST', 'test.fasta', 'alpha', 'MKT', 3, now()), "
    "('sp|P22222|BBB_TEST', 'test.fasta', 'beta', 'AAA', 3, now()), "
    "('sp|MISSING1|ZZZ_TEST', 'test.fasta', 'missing', 'CCC', 3, now())"
)


@pytest.fixture
def protein_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "uniprot.duckdb"
    conn = connect(db_path)
    init_database(conn)
    conn.execute(PROTEINS_SQL)
    conn.close()
    return db_path


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(HEADER + "\n" + "\n".join(ROWS) + "\n")
    return path


def test_download_snapshot_reuses_existing(tmp_path: Path) -> None:
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    dest = bulk / uniprot.SNAPSHOT_NAME
    dest.write_bytes(b"cached")
    assert uniprot.download_snapshot(bulk) == dest  # no network


class FakeResponse:
    """Minimal stand-in for a ``requests`` response."""

    def __init__(self, status_code: int, text: str, headers=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Session double that replays prepared responses and records calls."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url: str, params=None, timeout=None) -> FakeResponse:
        self.calls.append(params or {})
        response = self.responses.pop(0)
        if callable(response):
            self.responses.insert(0, response)  # reusable responder
            return response(params)
        return response


def test_staging_accepts_go_columns(tmp_path: Path) -> None:
    """The current 13-column snapshot must stage (regression: column count)."""
    path = tmp_path / "with_go.tsv"
    path.write_text(HEADER_WITH_GO + "\n" + "\n".join(ROWS_WITH_GO) + "\n")

    conn = connect(":memory:")
    init_database(conn)
    try:
        assert uniprot._build_staging_table(conn, [str(path)]) == 2
        assert conn.execute(
            f'SELECT "Function [CC]" FROM {uniprot.STAGING_TABLE} '
            'WHERE "Entry" = \'P11111\''
        ).fetchone()[0] == "FUNCTION: first function."
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


def test_annotate_database(protein_db: Path, snapshot: Path, monkeypatch) -> None:
    monkeypatch.setattr(uniprot, "download_snapshot", lambda *a, **k: snapshot)

    summary = uniprot.annotate_database(db_path=protein_db, source="bulk")

    assert summary["total_local_proteins"] == 3
    assert summary["annotated"] == 2
    assert summary["missing_function"] == 1  # P22222 has an empty function
    assert summary["not_found"] == 1  # MISSING1 has no staged row

    conn = connect(protein_db)
    try:
        row = conn.execute(
            "SELECT uniprot_id, function, organism, status FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST'"
        ).fetchone()
        assert row == ("AAA_TEST", "FUNCTION: first function.", "Homo sapiens", "fetched")
        status = conn.execute(
            "SELECT status FROM uniprot_annotations WHERE fasta_id = 'sp|P22222|BBB_TEST'"
        ).fetchone()[0]
        assert status == "missing_function"
        not_found = conn.execute(
            "SELECT status FROM uniprot_annotations WHERE fasta_id = 'sp|MISSING1|ZZZ_TEST'"
        ).fetchone()
        assert not_found == ("not_found",)
    finally:
        conn.close()



def test_annotate_fans_out_to_duplicate_rows(
    protein_db: Path, snapshot: Path, monkeypatch
) -> None:
    conn = connect(protein_db)
    conn.execute(
        "INSERT INTO proteins (fasta_id, source_file, description, sequence, "
        "sequence_length, created_at) VALUES "
        "('sp|P11111|AAA_TEST', 'other.fasta', 'alpha2', 'MKT', 3, now())"
    )
    conn.close()
    monkeypatch.setattr(uniprot, "download_snapshot", lambda *a, **k: snapshot)

    summary = uniprot.annotate_database(db_path=protein_db, source="bulk")
    assert summary["annotated"] == 3

    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST'"
        ).fetchone()[0] == 2
    finally:
        conn.close()


API_ROWS = {"P11111": ROWS_WITH_GO[0], "P22222": ROWS_WITH_GO[1]}


def _api_fetcher(rows: dict[str, str]):
    """Build a ``fetch_tsv`` replacement answering only what was requested."""

    def fetch(accessions, **kwargs):
        return HEADER_WITH_GO, [rows[a] for a in accessions if a in rows], []

    return fetch


def test_select_stale_rows_filters(protein_db: Path) -> None:
    conn = connect(protein_db)
    try:
        assert [row[2] for row in uniprot.select_stale_rows(conn)] == [
            "MISSING1",
            "P11111",
            "P22222",
        ]
        assert len(uniprot.select_stale_rows(conn, only_missing=True)) == 3

        conn.execute(
            "INSERT INTO uniprot_annotations (fasta_id, source_file, status, retrieved_at) "
            "VALUES ('sp|P11111|AAA_TEST', 'test.fasta', 'fetched', "
            "timezone('UTC', now()))"
        )
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


def test_api_annotation_writes_rows_and_marks_not_found(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(uniprot, "fetch_tsv", _api_fetcher(API_ROWS))

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
            "WHERE fasta_id = 'sp|P11111|AAA_TEST'"
        ).fetchone() == ("AAA_TEST", "FUNCTION: first function.", "fetched")
        assert conn.execute(
            "SELECT status FROM uniprot_annotations WHERE fasta_id = 'sp|MISSING1|ZZZ_TEST'"
        ).fetchone() == ("not_found",)
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 3
    finally:
        conn.close()


def test_api_second_run_requests_nothing(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(uniprot, "fetch_tsv", _api_fetcher(API_ROWS))
    kwargs = {
        "db_path": protein_db,
        "cache_dir": tmp_path / "cache",
        "source": "api",
        "only_missing": True,
    }
    uniprot.annotate_database(**kwargs)

    summary = uniprot.annotate_database(**kwargs)

    assert summary["accessions"] == 0
    assert summary["annotated"] == 0
    assert summary["http_requests"] == 0


def test_api_updates_existing_annotation(
    protein_db: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(uniprot, "fetch_tsv", _api_fetcher(API_ROWS))
    uniprot.annotate_database(
        db_path=protein_db, cache_dir=tmp_path / "cache", source="api", only_missing=True
    )

    updated = dict(API_ROWS)
    updated["P11111"] = API_ROWS["P11111"].replace("first function", "second function")
    monkeypatch.setattr(uniprot, "fetch_tsv", _api_fetcher(updated))
    summary = uniprot.annotate_database(
        db_path=protein_db,
        cache_dir=tmp_path / "cache",
        source="api",
        older_than_days=0,
    )

    assert summary["annotated"] == 2
    conn = connect(protein_db)
    try:
        assert conn.execute(
            "SELECT function FROM uniprot_annotations "
            "WHERE fasta_id = 'sp|P11111|AAA_TEST'"
        ).fetchone()[0].startswith("FUNCTION: second function")
        assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 3
    finally:
        conn.close()


def test_api_chunks_requests(protein_db: Path, tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    fetcher = _api_fetcher(API_ROWS)

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


def test_request_tsv_retries_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: sleeps.append(seconds))
    session = FakeSession(
        [
            FakeResponse(429, "slow down", {"Retry-After": "2"}),
            FakeResponse(200, HEADER_WITH_GO + "\n" + ROWS_WITH_GO[0] + "\n"),
        ]
    )

    text = uniprot._request_tsv(
        ["P11111"], session, max_retries=1, base_delay=1.0, max_delay=30.0
    )

    assert "P11111" in text
    assert sleeps == [2.0]
    assert len(session.calls) == 2


def test_fetch_tsv_bisects_on_rejected_accession(monkeypatch) -> None:
    monkeypatch.setattr(uniprot.time, "sleep", lambda seconds: None)
    good = {"P11111": ROWS_WITH_GO[0], "P22222": ROWS_WITH_GO[1], "P33333": ROWS_WITH_GO[0]}

    def responder(params):
        accessions = params["accessions"].split(",")
        if "BADACC" in accessions:
            return FakeResponse(400, "Accession 'BADACC' has invalid format.")
        return FakeResponse(
            200, HEADER_WITH_GO + "\n" + "\n".join(good[a] for a in accessions)
        )

    session = FakeSession([responder])
    header, rows, rejected = uniprot.fetch_tsv(
        ["P11111", "BADACC", "P22222", "P33333"], session=session, max_retries=0
    )

    assert header == HEADER_WITH_GO
    assert rejected == ["BADACC"]
    assert len(rows) == 3


def test_fetch_tsv_rejects_malformed_without_request() -> None:
    session = FakeSession([])  # any request would raise IndexError

    header, rows, rejected = uniprot.fetch_tsv(["not an accession"], session=session)

    assert header is None and rows == []
    assert rejected == ["not an accession"]
    assert session.calls == []


def test_filters_require_api_source(protein_db: Path) -> None:
    with pytest.raises(ValueError, match="require source='api'"):
        uniprot.annotate_database(db_path=protein_db, source="bulk", only_missing=True)
    with pytest.raises(ValueError, match="chunk_size"):
        uniprot.annotate_database(db_path=protein_db, source="api", chunk_size=1001)
    with pytest.raises(ValueError, match="Unknown source"):
        uniprot.annotate_database(db_path=protein_db, source="nope")


