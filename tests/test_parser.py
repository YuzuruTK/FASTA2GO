"""Tests for the FASTA parser module."""

from pathlib import Path

import pytest

from protein_database.parser.fasta_parser import (
    FastaRecord,
    count_fasta_entries,
    parse_fasta,
)


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


def test_count_fasta_entries(fasta_file: Path) -> None:
    """Test counting FASTA entries."""
    assert count_fasta_entries(fasta_file) == 3


def test_parse_fasta_records(fasta_file: Path) -> None:
    """Test parsing FASTA records."""
    records = list(parse_fasta(fasta_file))
    assert len(records) == 3

    # First record
    record = records[0]
    assert record.fasta_id == "sp|P69905|HBA_HUMAN"
    assert record.description == "Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2"
    assert record.sequence_length == 142
    assert record.sequence.startswith("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF")
    assert record.sequence.endswith("STVLTSKYR")

    # Second record
    record = records[1]
    assert record.fasta_id == "sp|P68871|HBB_HUMAN"
    assert record.description == "Hemoglobin subunit beta OS=Homo sapiens OX=9606 GN=HBB PE=1 SV=2"
    assert record.sequence_length == 147

    # Third record
    record = records[2]
    assert record.fasta_id == "sp|Q6GZX4|001R_FRG3G"
    assert record.sequence_length == 60


def test_parse_fasta_empty_file(tmp_path: Path) -> None:
    """Test parsing an empty FASTA file."""
    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("")
    records = list(parse_fasta(empty_file))
    assert len(records) == 0


def test_parse_fasta_single_record(tmp_path: Path) -> None:
    """Test parsing a single-record FASTA file."""
    single_file = tmp_path / "single.fasta"
    single_file.write_text(">test_protein Some description\nACDEFGHIKLMNPQRSTVWY\n")
    records = list(parse_fasta(single_file))
    assert len(records) == 1
    assert records[0].fasta_id == "test_protein"
    assert records[0].description == "Some description"
    assert records[0].sequence == "ACDEFGHIKLMNPQRSTVWY"
    assert records[0].sequence_length == 20


def test_parse_fasta_no_description(tmp_path: Path) -> None:
    """Test parsing a FASTA record without a description."""
    no_desc_file = tmp_path / "no_desc.fasta"
    no_desc_file.write_text(">just_id\nACDEFG\n")
    records = list(parse_fasta(no_desc_file))
    assert len(records) == 1
    assert records[0].fasta_id == "just_id"
    assert records[0].description == ""
    assert records[0].sequence == "ACDEFG"


def test_fasta_record_dataclass() -> None:
    """Test the FastaRecord dataclass."""
    record = FastaRecord(
        fasta_id="test",
        description="desc",
        sequence="ACDEFG",
    )
    assert record.sequence_length == 6