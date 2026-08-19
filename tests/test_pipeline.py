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

    summary = process_fasta_file(fasta_path=fasta_file, db_path=db_path, batch_size=2)

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

    first = process_fasta_file(fasta_path=fasta_file, db_path=db_path)
    assert first["processed"] == 3

    second = process_fasta_file(fasta_path=fasta_file, db_path=db_path)
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
        fasta_path=fasta_file, db_path=db_path, overwrite=True
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