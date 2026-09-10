"""Tests for the DuckDB processing pipeline."""

from pathlib import Path

import pytest

from protein_database.database.connection import connect, init_database
from protein_database.pipeline import process_fasta_file

SAMPLE_FASTA = """>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF
DLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRV
DPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR

>sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens OX=9606 GN=HBB PE=1 SV=2
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGD
LSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDK
LHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAH
KYH

>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 OX=654924 GN=FV3-001R PE=4 SV=1
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECP
KAPVEWNNPPS
"""


@pytest.fixture
def fasta_file(tmp_path: Path) -> Path:
    """Create a temporary FASTA file for testing."""
    file_path = tmp_path / "test.fasta"
    file_path.write_text(SAMPLE_FASTA)
    return file_path


def test_process_fasta_file_populates_database(tmp_path: Path, fasta_file: Path) -> None:
    """Test processing a FASTA file fills all tables correctly."""
    db_path = tmp_path / "test.duckdb"

    summary = process_fasta_file(
        fasta_path=fasta_file, db_path=db_path, batch_size=2,
        csv_cache_dir=tmp_path / "csv_cache",
    )

    assert summary["processed"] == 3
    assert summary["skipped"] == 0
    assert summary["errors"] == 0

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM proteins").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM sequence_features").fetchone()[0] == 3
        assert conn.execute(
            "SELECT count(*) FROM amino_acid_composition"
        ).fetchone()[0] == 3

        # Verify a known protein row
        row = conn.execute(
            "SELECT fasta_id, sequence_length, source_file "
            "FROM proteins WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
        ).fetchone()
        assert row[1] == 142
        assert row[2] == fasta_file.name

        # Verify a feature was computed
        feat = conn.execute(
            "SELECT molecular_weight FROM sequence_features "
            "WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
        ).fetchone()[0]
        assert feat is not None and feat > 0
    finally:
        conn.close()


def test_process_fasta_file_skips_duplicates(tmp_path: Path, fasta_file: Path) -> None:
    """Test that re-processing skips proteins already in the database."""
    db_path = tmp_path / "test.duckdb"

    first = process_fasta_file(
        fasta_path=fasta_file, db_path=db_path, csv_cache_dir=tmp_path / "csv_cache"
    )
    assert first["processed"] == 3

    second = process_fasta_file(
        fasta_path=fasta_file, db_path=db_path, csv_cache_dir=tmp_path / "csv_cache"
    )
    assert second["processed"] == 0
    assert second["skipped"] == 3

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM proteins").fetchone()[0] == 3
    finally:
        conn.close()


def test_process_fasta_file_overwrite(tmp_path: Path, fasta_file: Path) -> None:
    """Test that overwrite truncates and re-inserts."""
    db_path = tmp_path / "test.duckdb"
    # Add an extra record not in the fasta file to verify it's removed by overwrite
    conn = connect(db_path)
    init_database(conn)
    conn.execute(
        "INSERT INTO proteins (fasta_id, source_file, sequence, sequence_length, created_at) "
        "VALUES ('stale', 'other.fasta', 'AAAA', 4, now())"
    )
    conn.close()

    summary = process_fasta_file(
        fasta_path=fasta_file, db_path=db_path, overwrite=True,
        csv_cache_dir=tmp_path / "csv_cache",
    )
    assert summary["processed"] == 3
    assert summary["skipped"] == 0

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM proteins").fetchone()[0] == 3
        stale = conn.execute(
            "SELECT count(*) FROM proteins WHERE fasta_id = 'stale'"
        ).fetchone()[0]
        assert stale == 0
    finally:
        conn.close()


def test_process_fasta_file_round_trips_special_characters(tmp_path: Path) -> None:
    """Test that commas/quotes/newlines in descriptions survive CSV staging."""
    fasta = tmp_path / "special.fasta"
    fasta.write_text(
        '>id1 description with, a comma and "quotes"\n'
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF\n"
        ">id2\n"
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGD\n"
    )
    db_path = tmp_path / "special.duckdb"
    cache_dir = tmp_path / "csv_cache"

    summary = process_fasta_file(
        fasta_path=fasta, db_path=db_path, csv_cache_dir=cache_dir
    )
    assert summary["processed"] == 2
    assert summary["errors"] == 0

    conn = connect(db_path)
    try:
        desc = conn.execute(
            "SELECT description FROM proteins WHERE fasta_id = 'id1'"
        ).fetchone()[0]
        assert desc == 'description with, a comma and "quotes"'

        # Records without a description line store an empty CSV field, which
        # DuckDB reads back as SQL NULL.
        null_desc = conn.execute(
            "SELECT description FROM proteins WHERE fasta_id = 'id2'"
        ).fetchone()[0]
        assert null_desc is None
    finally:
        conn.close()

    # Staging CSVs are removed after a successful import.
    assert list(cache_dir.glob("*.csv")) == []


def test_insert_failure_keeps_staged_csv(tmp_path: Path) -> None:
    """Test that a failed CSV import leaves the staged file in cache."""
    from protein_database.pipeline import _insert_many

    db_path = tmp_path / "fail.duckdb"
    cache_dir = tmp_path / "csv_cache"
    conn = connect(db_path)
    init_database(conn)
    conn.execute("CREATE TABLE not_null_t (a VARCHAR NOT NULL, b INTEGER NOT NULL)")
    try:
        # ``b`` is NOT NULL; the empty CSV field reads back as NULL and the
        # native import must fail, leaving the staged CSV for debugging.
        with pytest.raises(Exception):
            _insert_many(
                conn,
                "not_null_t",
                ["a", "b"],
                [["id1", None]],
                cache_dir=cache_dir,
            )
    finally:
        conn.close()

    leftover = list(cache_dir.glob("*.csv"))
    assert len(leftover) == 1