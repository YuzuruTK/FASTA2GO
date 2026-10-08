"""Tests for UniProt snapshot download and native annotation ingestion."""

import gzip
from pathlib import Path

import pytest

import protein_database.uniprot as uniprot
from protein_database.db import connect, init_database

HEADER = (
    "Entry\tEntry Name\tProtein names\tGene Names\tFunction [CC]\t"
    "Subcellular location [CC]\tProtein existence\tOrganism\tKeywords"
)

ROWS = [
    "P11111\tAAA_TEST\tTest protein alpha\tGENA\tFUNCTION: first function.\t"
    "Cytoplasm\tEvidence at protein level\tHomo sapiens\tReference proteome",
    "P22222\tBBB_TEST\tTest protein beta\tGENB\t\t\t"
    "Inferred from homology\tMus musculus\tTransferase",
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
