"""Tests for the FASTA parser module."""

from pathlib import Path

from protein_database.fasta import FastaRecord, parse_fasta

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


def _write(tmp_path: Path, text: str, name: str = "test.fasta") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parse_fasta_records(tmp_path: Path) -> None:
    records = list(parse_fasta(_write(tmp_path, SAMPLE_FASTA)))
    assert len(records) == 3

    assert records[0].fasta_id == "sp|P69905|HBA_HUMAN"
    assert records[0].description == (
        "Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2"
    )
    assert records[0].sequence_length == 142
    assert records[0].sequence.startswith("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF")
    assert records[0].sequence.endswith("STVLTSKYR")

    assert records[1].fasta_id == "sp|P68871|HBB_HUMAN"
    assert records[1].sequence_length == 147
    assert records[2].fasta_id == "sp|Q6GZX4|001R_FRG3G"
    assert records[2].sequence_length == 60


def test_parse_fasta_empty_file(tmp_path: Path) -> None:
    assert list(parse_fasta(_write(tmp_path, "", "empty.fasta"))) == []


def test_parse_fasta_single_record(tmp_path: Path) -> None:
    records = list(
        parse_fasta(
            _write(tmp_path, ">test_protein Some description\nACDEFGHIKLMNPQRSTVWY\n")
        )
    )
    assert len(records) == 1
    assert records[0].fasta_id == "test_protein"
    assert records[0].description == "Some description"
    assert records[0].sequence == "ACDEFGHIKLMNPQRSTVWY"


def test_parse_fasta_no_description(tmp_path: Path) -> None:
    records = list(parse_fasta(_write(tmp_path, ">just_id\nACDEFG\n")))
    assert records[0].description == ""
    assert records[0].sequence == "ACDEFG"


def test_fasta_record_dataclass() -> None:
    record = FastaRecord("id", "desc", "ACDEFG")
    assert record.sequence_length == 6
