"""Tests for UniProt accession extraction and deduplication."""

from protein_database.uniprot.accession import (
    extract_accession,
    extract_accessions,
    group_accessions_by_local_rows,
)


def test_extract_accession_sp() -> None:
    """Extract accession from a reviewed (sp) FASTA identifier."""
    assert extract_accession("sp|P69905|HBA_HUMAN") == "P69905"


def test_extract_accession_tr() -> None:
    """Extract accession from an unreviewed (tr) FASTA identifier."""
    assert extract_accession("tr|A0A0C5B5G6|MOTSC_HUMAN") == "A0A0C5B5G6"


def test_extract_accession_uppercases() -> None:
    """Accessions are normalized to uppercase."""
    assert extract_accession("sp|p69905|HBA_HUMAN") == "P69905"


def test_extract_accession_non_uniprot_returns_none() -> None:
    """Non-UniProt identifiers return None."""
    assert extract_accession("gi|1234|lcl|foo") is None
    assert extract_accession("P69905") is None
    assert extract_accession("") is None
    assert extract_accession("sp||HBA_HUMAN") is None


def test_extract_accessions_dedups() -> None:
    """Duplicate accessions are collapsed."""
    ids = [
        "sp|P69905|HBA_HUMAN",
        "tr|A0A0C5B5G6|MOTSC_HUMAN",
        "sp|P69905|HBA_HUMAN",  # dup
        "sp|P68871|HBB_HUMAN",
    ]
    assert extract_accessions(ids) == ["A0A0C5B5G6", "P68871", "P69905"]


def test_group_accessions_by_local_rows() -> None:
    """One accession maps to multiple local protein rows."""
    rows = [
        ("sp|P69905|HBA_HUMAN", "file1.fasta", "x"),
        ("sp|P69905|HBA_HUMAN", "file2.fasta", "x"),
        ("sp|P68871|HBB_HUMAN", "file1.fasta", "x"),
    ]
    mapping = group_accessions_by_local_rows(rows)
    assert set(mapping["P69905"]) == {
        ("sp|P69905|HBA_HUMAN", "file1.fasta"),
        ("sp|P69905|HBA_HUMAN", "file2.fasta"),
    }
    assert set(mapping["P68871"]) == {("sp|P68871|HBB_HUMAN", "file1.fasta")}