"""Tests for the DuckDB processing pipeline."""

from pathlib import Path

import pytest

from protein_database.db import bulk_insert, connect, init_database
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
    path = tmp_path / "test.fasta"
    path.write_text(SAMPLE_FASTA)
    return path


def test_process_fasta_file_populates_database(tmp_path: Path, fasta_file: Path) -> None:
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
        for table in ("proteins", "sequence_features", "amino_acid_composition"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 3
        row = conn.execute(
            "SELECT sequence_length, source_file FROM proteins "
            "WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
        ).fetchone()
        assert row == (142, fasta_file.name)
        mw = conn.execute(
            "SELECT molecular_weight FROM sequence_features "
            "WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
        ).fetchone()[0]
        assert mw is not None and mw > 0
    finally:
        conn.close()


def test_process_fasta_file_skips_duplicates(tmp_path: Path, fasta_file: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    cache = tmp_path / "csv_cache"
    assert process_fasta_file(fasta_file, db_path, csv_cache_dir=cache)["processed"] == 3
    second = process_fasta_file(fasta_file, db_path, csv_cache_dir=cache)
    assert second["processed"] == 0
    assert second["skipped"] == 3


def test_process_fasta_file_overwrite(tmp_path: Path, fasta_file: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    conn = connect(db_path)
    init_database(conn)
    conn.execute(
        "INSERT INTO proteins (fasta_id, source_file, sequence, sequence_length, created_at) "
        "VALUES ('stale', 'other.fasta', 'AAAA', 4, now())"
    )
    conn.close()

    summary = process_fasta_file(
        fasta_file, db_path, overwrite=True, csv_cache_dir=tmp_path / "csv_cache"
    )
    assert summary["processed"] == 3
    conn = connect(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM proteins").fetchone()[0] == 3
        assert conn.execute(
            "SELECT count(*) FROM proteins WHERE fasta_id = 'stale'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_process_fasta_file_round_trips_special_characters(tmp_path: Path) -> None:
    fasta = tmp_path / "special.fasta"
    fasta.write_text(
        '>id1 description with, a comma and "quotes"\n'
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF\n"
        ">id2\n"
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGD\n"
    )
    db_path = tmp_path / "special.duckdb"
    cache_dir = tmp_path / "csv_cache"

    summary = process_fasta_file(fasta, db_path, csv_cache_dir=cache_dir)
    assert summary["processed"] == 2
    assert summary["errors"] == 0

    conn = connect(db_path)
    try:
        assert conn.execute(
            "SELECT description FROM proteins WHERE fasta_id = 'id1'"
        ).fetchone()[0] == 'description with, a comma and "quotes"'
        # An empty description becomes an empty CSV field -> SQL NULL.
        assert conn.execute(
            "SELECT description FROM proteins WHERE fasta_id = 'id2'"
        ).fetchone()[0] is None
    finally:
        conn.close()
    assert list(cache_dir.glob("*.csv")) == []


def test_bulk_insert_failure_keeps_staged_csv(tmp_path: Path) -> None:
    conn = connect(tmp_path / "fail.duckdb")
    cache_dir = tmp_path / "csv_cache"
    try:
        init_database(conn)
        conn.execute("CREATE TABLE not_null_t (a VARCHAR NOT NULL, b INTEGER NOT NULL)")
        with pytest.raises(Exception):
            bulk_insert(conn, "not_null_t", ["a", "b"], [["id1", None]], cache_dir)
    finally:
        conn.close()
    assert len(list(cache_dir.glob("*.csv"))) == 1
